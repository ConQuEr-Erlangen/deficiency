(* ::Package:: *)

(*Orthogonal Polynomials of first and second kind. Output is list X_0...X_Nmax*)
PFn[z_,a_,b_,Nmax_]:=RecurrenceTable[{c[n+1]==Inverse[a[n]] . ((z IdentityMatrix[Length[a[0]]]-b[n]) . c[n]-ConjugateTranspose[a[n-1]] . c[n-1]),c[-1]==ConstantArray[0,{Length[a[0]],Length[a[0]]}],c[0]==IdentityMatrix[Length[a[0]]]},c,{n,0,Nmax}];
QFn[z_,a_,b_,Nmax_]:=RecurrenceTable[{c[n+1]==Inverse[a[n]] . ((z IdentityMatrix[Length[a[0]]]-b[n]) . c[n]-ConjugateTranspose[a[n-1]] . c[n-1]),c[0]==ConstantArray[0,{Length[a[0]],Length[a[0]]}],c[1]==Inverse[a[0]]},c,{n,0,Nmax}];


(*Weyl radius operators R_0...R_n generated from the orthgonal polynomials of first kind*)
weylRadiusOp[z_,pfn_]:=Inverse /@ (2 Abs[Im[z]]Accumulate[(ConjugateTranspose[#] . #)& /@ pfn]);
(*The associated radii*)
weylRadius[z_,pfn_]:=Eigenvalues /@ weylRadiusOp[z,pfn];


(* Blocks specifically for higher order squeezing *)
\[Beta][n_,k_]:=Sqrt[Pochhammer[n+1,k]];
ABlock[n_,k_]:=DiagonalMatrix[Table[\[Beta][n k+i,k],{i,0,k-1}]];
BBlockDiag[n_,k_,f_]:=DiagonalMatrix[Table[f[n k + i],{i,0,k-1}]];


(*Blocks for general polynomials, generated from the coefficient matrix*)
\[Beta][n_,k_,l_]:=Sqrt[Pochhammer[n-l+1,l]Pochhammer[n-l+1,k]];
ABlock[n_,maxShift_,clist_]:=Table[Sum[clist[[l-maxShift+i-j+1,l+1]]\[Beta][maxShift n+i,l,l-maxShift+i-j],{l,maxShift-i+j,Length[clist]-1}],{i,0,maxShift-1},{j,0,maxShift-1}];
BBlock[n_,maxShift_,clist_]:=Table[Sum[clist[[l+i-j+1,l+1]]\[Beta][maxShift n+j,l+i-j,l],{l,Max[0,j-i],Min[Length[clist]-1,Length[clist]+j-i-1]}],{i,0,maxShift-1},{j,0,maxShift-1}];


(*Implement bosonic algebra*)
bosonicAlgabra=WeylAlgebra[{{adag},{a}}];

(*Helper functions for taking the formal adjoint*)
ClearAll[HermConj];
HermConj[expr_Plus] := HermConj /@ expr                                  (* linear *)
HermConj[expr_Times] := HermConj /@ expr                                 (* conjugate scalars, dagger the operator part *)
HermConj[expr_NonCommutativeMultiply] := Reverse[HermConj /@ expr]       (* (AB)\[Dagger] = B\[Dagger]A\[Dagger] *)
HermConj[GeneralizedPower[NonCommutativeMultiply, x_, n_]] := GeneralizedPower[NonCommutativeMultiply, HermConj[x], n]          (* (x^n)\[Dagger] = (x\[Dagger])^n *)
HermConj[a] := adag
HermConj[adag] := a
HermConj[x]:=x
HermConj[q]:=q
HermConj[p]:=p
HermConj[c_] /; FreeQ[c, a | adag] := Conjugate[c]                  (* scalars *)
(*Helper function to obtain the Hermitian part of a polynomial*)
Hermitianify[poly_]:=1/2 NonCommutativeExpand[poly+HermConj[poly]];

(*Weyl-ordered operator of a classical polynomial in p and either x or q
  If factors a or adag appear, a warning is returned.
*)
WeylSymmetrize::warningladder="warning: WeylSymmetrize should only be applied onto polynomials in x and p. Returning original polynomial";
WeylSymmetrize::warningxq="warning: Use either x or q. Returning original polynomial";
WeylSymmetrize[poly_]:=Module[{e,used,u,v,word},
	e=Expand[poly /. {GeneralizedPower[NonCommutativeMultiply,y_,k_]:>y^k, NonCommutativeMultiply->Times}];
	used=Select[{a,adag,x,q,p},!FreeQ[e,#]&];
	If[MemberQ[used,a]||MemberQ[used,adag],
		Message[WeylSymmetrize::warningladder];
		Return[poly];
	];
	If[MemberQ[used,q] && MemberQ[used,x],
		Message[WeylSymmetrize::warningxq];
		Return[poly];
	];
	u=If[MemberQ[used,q]&&!MemberQ[used,x],q,x];
	v=p;
	word[pos_,1]:=If[pos=={},v,u];
	word[pos_,n_]:=NonCommutativeMultiply@@ReplacePart[ConstantArray[v,n],List/@pos->u];
	Total[Function[{kl,c},
		If[Total[kl]==0,
			c,
			c Mean[
				word[#,Total[kl]]&/@Subsets[Range[Total[kl]],{kl[[1]]}]
			]
		]
	]@@@CoefficientRules[e,{u,v}]]
];


(*Helper function to turn a polynomial in x and p into one in a and adag*)
toBosonicPoly[poly_]:= poly/.{
      x -> (a + adag)/Sqrt[2],
      q -> (a + adag)/Sqrt[2],
      p -> (a - adag)/(Sqrt[2] I)
    };


(*Turns a polynomial in a and adagger into a coefficient matrix*)
Coefficients[poly_]:=Module[{expr,commutingexpr},
	expr=NonCommutativeExpand[poly,bosonicAlgabra];
	commutingexpr=expr /. {GeneralizedPower[NonCommutativeMultiply,x_,k_]:>x^k, NonCommutativeMultiply->Times};
	CoefficientList[commutingexpr,{adag,a}]
];


(*helper functions acting on on the coefficient matrix*)
(*maximum photon number shift induced by the polynomial*)
maxShift[clist_]:=Max[Abs[Subtract @@@ Position[clist,_?(#!=0&),2]]];
(*maximum power of the polynomial*)
maxPower[clist_]:=Max [Plus @@@(Position[clist, _?(#!=0&),2]-1)];
(*calculates for which n all diagonals are in the asymptotic regime, and such that the dominant diagonal dominates*)
getDominantN[coeffs_]:=Module[{nzPos,byDiagSorted,maxPos,maxDiagPos,maxForDiags,maxForTotal},
	nzPos=Position[coeffs,_?(#!=0&),2];
	byDiagSorted=Map[SortBy[#,First]&,GroupBy[nzPos,Subtract@@Reverse[#]&]];
	maxPos = Map[Last,byDiagSorted];
	maxDiagPos = Last@SortBy[maxPos,Total];
	maxForDiags = Max@@KeyValueMap[Function[{diag,positions},If[Length[positions]==1,1,Ceiling[N@Abs[Extract[coeffs,#]/Extract[coeffs,maxPos[diag]]]^(2/(Total@maxPos[diag]-Total@#))]& /@ positions[[;;-2]]]],byDiagSorted];
	maxForTotal = If[Length[maxPos]==1,1,Max@Map[If[Total@maxDiagPos==Total@#,1,Ceiling[N@Abs[Extract[coeffs,#]/Extract[coeffs,maxDiagPos]]^(2/(Total@maxDiagPos-Total@#))]]&,SortBy[maxPos,Total][[;;-2]]]];
	Max[maxForDiags,maxForTotal]
];

(*rescales polynomial with a scalar, such that the coefficient of the highest shift is 1*)
getRescaling[coeffs_]:=Module[{nzPos,byDiagSorted},
	nzPos=Position[coeffs,_?(#!=0&),2];
	byDiagSorted=Map[SortBy[#,First]&,GroupBy[nzPos,Subtract@@Reverse[#]&]];
	1/Abs[Extract[coeffs,Last@byDiagSorted[Max[Keys[byDiagSorted]]]]]
];
(*checks whether the two rescaling are necessary*)
ScaleTest[coeffs_]:= (getDominantN[coeffs]>10) || getRescaling[coeffs]>10 || 1/getRescaling[coeffs]>10;


(*Calculate the radius operators R_0...R_Nmax numerically with precision Precision*)
(*Use StartIndex to skip block A_n with 0 determinant*)
(*Use AutoScale to automatically apply the rescaling in N and prefactor from above*)
RadiusOperators::precision = "Error: Precision of polynomial prefactors smaller than requested precision";
RadiusOperators::hermitian = "Error: Coefficient list has to be hermitian";
RadiusOperators::rescaleWarning = "Warning: Rescaling coefficients is recommended. Activate AutoScale->True";
RadiusOperators::rescaling = "Warning: Rescaling coefficients and Length";
Options[RadiusOperators]={
	Precision->MachinePrecision,
	StartIndex->0,
	AutoScale->False,
	Warnings->True
};
RadiusOperators[poly_,z_,Nmax_,OptionsPattern[]]:=Module[{mcoefflist,md,optPrec,optStartindex,mNmax},
	mcoefflist=Coefficients[toBosonicPoly@poly];
	If[OptionValue[Warnings] && !HermitianMatrixQ[mcoefflist],
		Message[RadiusOperators::hermitian];
	];
	If[OptionValue[Warnings] && ScaleTest[mcoefflist] && !OptionValue[AutoScale],
		Message[RadiusOperators::rescaleWarning];
	];
	If[OptionValue[AutoScale],
		Message[RadiusOperators::rescaling];
	];
	mNmax = If[OptionValue[AutoScale],
		getDominantN[mcoefflist] Nmax,
		Nmax
	];
	mcoefflist = If[OptionValue[AutoScale],
		getRescaling[mcoefflist] mcoefflist,
		mcoefflist
	];
		
	
	md=maxShift[mcoefflist];
	optPrec=OptionValue[Precision];
	If[Precision[mcoefflist] < optPrec,
		Message[DeficiencyPlot::precision];
		Return[$Failed]
	];
	optStartindex = OptionValue[StartIndex];
	weylRadiusOp[z,PFn[z, N[ABlock[#+optStartindex,md,mcoefflist],optPrec]&,N[BBlock[#+optStartindex,md,mcoefflist],optPrec] &,mNmax]]
];
(*Calulate the bands \[Rho]_0...\[Rho]_Nmax up to Precision Precision*)
(*Use StartIndex to skip block A_n with 0 determinant*)
(*Use AutoScale to automatically apply the rescaling in N and prefactor from above*)
(*Plot using ListPlot[#] or raabeList /@ #*) 
Options[DeficiencyPlot]={
	Precision->MachinePrecision,
	StartIndex->0,
	AutoScale->False,
	Warnings->True	
};
DeficiencyPlot[poly_,z_,Nmax_,opts:OptionsPattern[]]:=Module[{radiusOps},
	radiusOps = RadiusOperators[poly,z,Nmax,opts];
	Transpose[Eigenvalues/@radiusOps]
];


(*Visualization of the Radius operators with respect to the Projected space P*)
VisualizeRadius[R_,P_]:=Module[{mP,mR},
	mP = Join[Re[#],Im[#]] & /@ P;
	mR = ArrayFlatten[{{Re[R],-Im[R]},{Im[R],Re[R]}}];
	mP . mR . Transpose[mP]
];


(*Check whether for a given A and Nmax, all A_0...A_Nmax have non-zero determinant*)
(*If not, a shift is needed*)
checkABlocks[a_,Nmax_]:=Max@Position[Abs @* Det @* a /@Range[0,Nmax],_?(#<1 &),1];


(* different test functions to check convergence
   returns booleans: {isConv,isZero,isNotZero}
   is allowed to return {isConv,False,False} when unsure
 *)
(*Naive default test, checking limit by thresholds*)
defaultTest[radia_List,nconv_,me0_,me1_]:={
	Max @ (radia[[-nconv;;]]-radia[[-1]]) < me0,
	Max @ radia[[-nconv;;]] < me0,
	Min @ radia[[-nconv;;]] > me1
};
(*Transform the radii into the discreteLogarithm*)
raabeList[a_List]:= Range[1,1+Length[a]-2]*(1-Rest[a]/Most[a]);
(*The Raabe Criterion is applied as discribed in the paper*)
raabeTest[radia_List,nconv_,me0_,me1_,mraabe_]:=Module[{isConv,isGreater1,isSmaller1,isNonZero,isZero,discreteLog},
	(*standard test*)
	isConv = Max @ (radia[[-nconv;;]]-radia[[-1]]) < me0;
	isNonZero = Min @ radia[[-nconv;;]] > me1;
	isZero = Max@radia[[-nconv;;]] < me0;
	(*use Rabe criterion to check whether actually 0 *)
	discreteLog = raabeList[radia];
	isGreater1 = Min @ discreteLog[[-nconv;;]] > mraabe;
	isSmaller1 = Max @ discreteLog[[-nconv;;]] < 1/mraabe;
	isZero = isZero && isGreater1;
	isNonZero = isNonZero && isSmaller1;
	isConv = isConv && (isZero != isNonZero);
	(*isSmaller1 -> series diverges, does not imply that sequences converges to finite number.*)
	{isConv,isZero,isNonZero}
];


(*  DeficiencyIndices of an operator described by a coefficient matrix at z
	returns <| state, n, #zeros|>
	 if state != 0, n lower bounds the deficiency indices and d-zeros upper bounds them
*)
(*Error Codes
 0: Success
-1: not converged yet
-2: radia increase -> precision too low
-3: badly  conditioned matrix -> precision too low
-4: deficiency indices do not sum up to d (convergence issues, might be recognized as converged but not as non-zero)
-5: imaginary radia -> precision too low
-6: Singular Matrix while calculating P[z] -> increase start index
-7: radia exactly zero -> precision too low
-8: error whilst calculating polynomials, but not due to determinant 0 -> precision too low
+1: Coefficient list does not have sufficient precision
+2: Coefficient list not hermitian
*)
DeficiencyIndices::precision = "Error: Precision of coefficient matrix smaller than algorithm precision";
DeficiencyIndices::hermitian = "Error: Coefficient list has to be hermitian.";
Options[DeficiencyIndices]={
	Precision->MachinePrecision,
	StartIndex->0,
	AlgorithmLength->50,
	Epsilon0->10^(-3),
	Epsilon1-> 10^(-2),
	RaabeRatio->2,
	ConvergenceTest->25,
	SimpleCarleman->True,
	Method->"Raabe"
};
DeficiencyIndices[z_?NumericQ,coefflist_?MatrixQ,OptionsPattern[]]:=Module[{
		optPrec,optStartindex,optLength,me0,me1,mraabe,optConvWindow,
		md,mppoly,mRadia,mconvresult,
		isImag,isNotConv,isNotMonotonic,isNotSum,
		count0,count1},
	If[!HermitianMatrixQ[coefflist],
		Message[DeficiencyIndices::hermitian];
		Return[<|"state"->2, "n"->0, "zeros"->0 |>]
	];
	md = maxShift[coefflist];
	(* diagonal operator *)
	If[md == 0,
		Return[<|"state"->0, "n"->0, "zeros"->0 |>]
	];
	
	(*check sufficient condition for Carleman*)
	If[OptionValue[SimpleCarleman] && maxPower[coefflist] <= 2,
		Return[<|"state"->0, "n"->0, "zeros"->md |>]
	];
	optPrec = OptionValue[Precision];
	If[Precision[coefflist] < optPrec,
		Message[DeficiencyIndices::precision];
		Return[<|"state"->1, "n"->0, "zeros"->0 |>]
	];
	
	optStartindex = OptionValue[StartIndex];
	optLength = OptionValue[AlgorithmLength];
	me0 = OptionValue[Epsilon0];
	me1 = OptionValue[Epsilon1];
	mraabe = OptionValue[RaabeRatio];
	optConvWindow = OptionValue[ConvergenceTest];
	mppoly = Quiet[Check[PFn[z, N[ABlock[#+optStartindex,md,coefflist],optPrec]&,N[BBlock[#+optStartindex,md,coefflist],optPrec] &,optLength],$Failed,{General::sing,General::infy}]];
	(*check (probably) too small start index*)
	If[mppoly === $Failed,
		With[{det0index = checkABlocks[N[ABlock[#+optStartindex,md,coefflist],optPrec]&,optLength]},
			If[det0index==-Infinity,
				Return[<|"state"->-8, "n"->0, "zeros"->0|>],
				Return[<|"state"->-6, "n"->0, "zeros"->0, "deltaShift"->det0index|>]
			]
		]
	];
	mRadia = Quiet[Check[Transpose[weylRadius[z,mppoly]],$Failed,{General::luc}]];
	(*check badly conditioned matrix errors*)
	If[mRadia === $Failed,
		Return[<|"state"->-3, "n"->0, "zeros"->0 |>]
	];
	(*ignore artificial imaginary parts*)
	isImag = !(Max @@ Im /@ mRadia < me0);
	mRadia = Re[mRadia];
	If[Min[mRadia]==0,
		Return[<|"state"->-7, "n"->0, "zeros"->0 |>]
	];
	
	isNotMonotonic = !(Max @@  Differences /@ mRadia <= 0);
	mconvresult = Switch[OptionValue[Method],
		"default", defaultTest[#,optConvWindow,me0,me1] & /@ mRadia,
		"Raabe", raabeTest[#,optConvWindow,me0,me1,mraabe] & /@ mRadia,
		_,defaultTest[#,optConvWindow,me0,me1] & /@ mRadia
		];
	isNotConv = !AllTrue[mconvresult[[All,1]],Identity];
	count0 = Count[mconvresult[[All,2]],True];
	count1 = Count[mconvresult[[All,3]],True];
	isNotSum = (count0+count1!=md);
	Which[
		isImag, <|"state"->-5, "n"->count1, "zeros"->count0 |>,
		isNotMonotonic,<|"state"->-2, "n"->count1, "zeros"->count0 |>,
		isNotConv, <|"state"->-1, "n"->count1, "zeros"->count0 |>,
		isNotSum,  <|"state"->-4, "n"->count1, "zeros"->count0 |>,
		True,  <|"state"-> 0, "n"->count1, "zeros"->count0 |>
	]
];
(*Wrappers for the above function*)
DeficiencyIndices[poly_,z_?NumericQ,opts:OptionsPattern[]]:=Module[{mcoefflist},
	mcoefflist=Coefficients[toBosonicPoly@poly];
	DeficiencyIndices[z,mcoefflist,opts]
];
DeficiencyIndices[poly_,opts:OptionsPattern[]]:={DeficiencyIndices[poly,I,opts],DeficiencyIndices[poly,-I,opts]};


(*Implement an adaptive strategy which automatically increases Precision,AlgorithmLength and StartIndex*)
Options[AdaptiveDeficiencyIndices]={
	Epsilon0->10^(-3),
	Epsilon1-> 10^(-2),
	RaabeRatio->2,
	ConvergenceTest->25,
	MinAlgorithmLength->50,
	MaxAlgorithmLength->10000,
	DeltaAlgorithmLength->100,
	MinPrecision->MachinePrecision,
	MaxPrecision->5000,
	MinStartIndex->0,
	SimpleCarleman->True,
	AdaptiveLength->True,
	AdaptiveRescaling->True,
	DebugPrint->False,
	Method->"Raabe"
};
AdaptiveDeficiencyIndices[z_?NumericQ,coefflist_?MatrixQ,opts:OptionsPattern[]]:=Module[
	{mIncreaseFn,mFilteredOptions,mcoefflist,mMinAlgorithmLength,mDeltaAlgorithmLength},
	(*filter options to pass through*)
	mFilteredOptions = FilterRules[{opts},Options[DeficiencyIndices]];
	
	mMinAlgorithmLength = If[OptionValue[AdaptiveLength],
		getDominantN[coefflist] OptionValue[MinAlgorithmLength],
		OptionValue[MinAlgorithmLength]
	];
	mDeltaAlgorithmLength = If[OptionValue[AdaptiveLength],
		getDominantN[coefflist] OptionValue[DeltaAlgorithmLength],
		OptionValue[DeltaAlgorithmLength]
	];
	mcoefflist = If[OptionValue[AdaptiveRescaling],
		getRescaling[coefflist] coefflist,
		coefflist
	];
	(* state = {result,precision,algorithmicdepth,startindex} *)
	mIncreaseFn=With[
		(*append precision, length, start to result list, without overwriting the new result*)
		{mresult = Merge[{
				DeficiencyIndices[z,mcoefflist,Precision->#["precision"],AlgorithmLength->#["length"],StartIndex->#["start"],Sequence @@ mFilteredOptions],
				#},First]
			},
		If[OptionValue[DebugPrint],
			Print[mresult];
		];
		Which[
			(* use append as non-mutating AssociateTo *)
			MemberQ[{-1,-4},mresult["state"]], Append[mresult,"length"->mresult["length"]+mDeltaAlgorithmLength],
			MemberQ[{-2,-3,-5,-7,-8},mresult["state"]], Append[mresult,"precision"->2 mresult["precision"]],
			MemberQ[{-6},mresult["state"]], Append[mresult,"start"->mresult["start"]+mresult["deltaShift"]],
			True,mresult
			]
		] & ;
	NestWhile[mIncreaseFn,<|"state"->-1,"precision"->OptionValue[MinPrecision],"length"->mMinAlgorithmLength,"start"->OptionValue[MinStartIndex]|>,#["state"]<0 && #["precision"]<=OptionValue[MaxPrecision] && #["length"] <= OptionValue[MaxAlgorithmLength] &]
];
(*Wrappers for the above function*)
AdaptiveDeficiencyIndices[poly_,z_?NumericQ,opts:OptionsPattern[]]:=Module[{mcoefflist},
	mcoefflist=Coefficients[toBosonicPoly@poly];
	AdaptiveDeficiencyIndices[z,mcoefflist,opts]
];
AdaptiveDeficiencyIndices[poly_,opts:OptionsPattern[]]:=Module[{mcoefflist},
	mcoefflist=Coefficients[toBosonicPoly@poly];
	If[FreeQ[mcoefflist,_Complex],
		With[{res = AdaptiveDeficiencyIndices[I,mcoefflist,opts]}, {res, res}],
		{AdaptiveDeficiencyIndices[I,mcoefflist,opts],AdaptiveDeficiencyIndices[-I,mcoefflist,opts]}
	]
];


(*Colored and easy to understand output*)

(*Display functions*)
adDisplay /: MakeBoxes[adDisplay, TraditionalForm] :=
  SuperscriptBox["a", "\[Dagger]"];
ncDisplay /: MakeBoxes[ncDisplay[args__], TraditionalForm] :=
  RowBox[
    MakeBoxes[#, TraditionalForm] & /@ {args}
  ];
HamiltonianDisplay[poly_] := Module[{pretty},
  pretty =
    poly /. HoldPattern[NonCommutativeMultiply[args__]] :>
      Module[{groups = Split[{args}]},
        ncDisplay @@
          (If[
              Length[#] == 1,
              First[#],
              First[#]^Length[#]
            ] & /@ groups)
      ];

  pretty = pretty /. adag -> adDisplay;

  TraditionalForm[pretty]
];


goodColor = RGBColor[48/255, 111/255, 29/255];
sosoColor = RGBColor[226/255, 121/255, 46/255];
badColor  = RGBColor[166/255, 43/255, 23/255];

SetAttributes[QuickDeficiencyIndices, HoldAll];

QuickDeficiencyIndices::error = "Error: Failed to compute deficiency indices. Use AdapativeDeficiencyIndices for detailed output and ListPlot@DeficiencyIndices for visual feedback";
QuickDeficiencyIndices::errorNotConverged = "Possible Problems: MaxAlgorithm Length is not sufficient; \!\(\*SubscriptBox[\(\[Epsilon]\), \(1\)]\) is too large";
QuickDeficiencyIndices::errorPrecision = "Possible Problems: MaxPrecision is not sufficient";
QuickDeficiencyIndices::errorCoefficients = "Error source: Coefficients do not have sufficient precision. Increase input precision";
QuickDeficiencyIndices::errorHermitian = "Error source: Input is not hermitian";

QuickDeficiencyIndices[poly_] := Module[
  {
    heldPoly,
    originalPoly,
    computationPoly,
    mres,
    np, nm,
    color, label
  },

  (* Keep exactly the Hamiltonian that was typed *)
  heldPoly = HoldComplete[poly];

  (* Since x,p,q have no global definitions, this is safe *)
  originalPoly = ReleaseHold[heldPoly];

  (* Convert x/q and p to a, adag if not already set *)
  computationPoly =
    originalPoly /. {
      x -> (a + adag)/Sqrt[2],
      q -> (a + adag)/Sqrt[2],
      p -> (a - adag)/(Sqrt[2] I)
    };

  mres = AdaptiveDeficiencyIndices[computationPoly];

  If[
    !AllTrue[mres, #["state"] == 0 &],

	Column[{Style[
          Row[{
            Style["H", Italic], " = ", HamiltonianDisplay[originalPoly]
          }], badColor, Bold
        ],
        Style[QuickDeficiencyIndices::error,badColor,Bold]
        ,
        Which[
            MemberQ[mres[[All,"state"]],+1], QuickDeficiencyIndices::errorCoefficients,
            MemberQ[mres[[All,"state"]],+2], QuickDeficiencyIndices::errorHermitian,
            MemberQ[mres[[All,"state"]],-1], QuickDeficiencyIndices::errorNotConverged,
            MemberQ[mres[[All,"state"]],-4], QuickDeficiencyIndices::errorNotConverged,
            True, QuickDeficiencyIndices::errorPrecision
        ]
        }],

    np = mres[[1, "n"]];
    nm = mres[[2, "n"]];

    {color, label} = Which[
      np == 0 && nm == 0, {goodColor, "good Hamiltonian"},
      np == nm, {sosoColor, "more physical information necessary"},
      True, {badColor, "bad Hamiltonian"}
    ];

    Column[
      {
        Style[
          Row[{
            Style["H", Italic], " = ", HamiltonianDisplay[originalPoly]
          }], color, Bold
        ],
        Row[{
          Subscript["n", "+"], " = ", np, ",   ", Subscript["n", "-"], " = ", nm, "   \[LongDash]   ", Style[label, color, Bold]
        }]
      },
      Spacings -> 0.7
    ]
  ]
]
