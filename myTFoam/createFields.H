Info<< "Reading field T\n" << endl;

volScalarField T
(
    IOobject
    (
        "T",
        runTime.timeName(),
        mesh,
        IOobject::MUST_READ,
        IOobject::AUTO_WRITE
    ),
    mesh
);

volScalarField d
(
    IOobject
    (
        "wallDistance",
        runTime.timeName(),
        mesh,
        IOobject::MUST_READ,
        IOobject::NO_WRITE
    ),
    mesh
);


Info<< "Reading field U\n" << endl;

volVectorField U
(
    IOobject
    (
        "U",
        runTime.timeName(),
        mesh,
        IOobject::MUST_READ,
        IOobject::AUTO_WRITE
    ),
    mesh
);

volScalarField strainMag
(
    IOobject
    (
        "S",
        runTime.timeName(),
        mesh,
        IOobject::NO_READ,
        IOobject::AUTO_WRITE
    ),
    ::sqrt(2.0)*mag(symm(fvc::grad(U)))
);

strainMag.write();

volScalarField omegaMag
(
    IOobject
    (
        "Omega",
        runTime.timeName(),
        mesh,
        IOobject::NO_READ,
        IOobject::AUTO_WRITE
    ),
    ::sqrt(2.0)*mag(skew(fvc::grad(U)))
);

omegaMag.write();

Info<< "Reading transportProperties\n" << endl;

IOdictionary transportProperties
(
    IOobject
    (
        "transportProperties",
        runTime.constant(),
        mesh,
        IOobject::MUST_READ_IF_MODIFIED,
        IOobject::NO_WRITE
    )
);


Info<< "Reading diffusivity DT\n" << endl;

dimensionedScalar DT("DT", dimViscosity, transportProperties);

dimensionedScalar delta("delta", dimLength, transportProperties);

// dimensionedScalar zeta("zeta", dimless/dimTime/sqr(dimVelocity), transportProperties);
dimensionedScalar zeta("zeta", dimless/dimLength, transportProperties);

dimensionedScalar gamma("gamma", dimless, transportProperties);

/*
dimensionedScalar gammaSNonlinear("gammaSNonlinear", dimTemperature/dimTime, transportProperties);

dimensionedScalar gammaSW("gammaSW", dimTemperature/dimTime, transportProperties);

dimensionedScalar gammaSWC("gammaSWC", dimTemperature/dimTime, transportProperties);
*/

dimensionedScalar S0("S0", dimless/dimTime, Foam::gMax(strainMag.internalField())); 

// dimensionedScalar S0("S0", Foam::gMax(strainMag.internalField())); 

Info << "Normalization constant for S: " << S0.value() << endl;

volScalarField mixingLength 
(
    IOobject
    (
        "mixingLength",
        runTime.timeName(),
        mesh,
        IOobject::NO_READ,
        IOobject::AUTO_WRITE
    ),
    min(0.41*d,0.09*delta) 
);
mixingLength.write();

/*
volScalarField PrSLinear
(
    IOobject
    (
        "Pr",
        runTime.timeName(),
        mesh,
        IOobject::NO_READ,
        IOobject::AUTO_WRITE
    ),
    strainMag/S0
);

PrSLinear.write();
*/

volScalarField Pr
(
    IOobject
    (
        "Pr",
        runTime.timeName(),
        mesh,
        IOobject::NO_READ,
        IOobject::AUTO_WRITE
    ),
    gamma * mixingLength * sqrt(T) * sqr(strainMag)
);

Pr.write();

/*
volScalarField PrSNonlinear
(
    IOobject
    (
        "PrSNonlinear",
        runTime.timeName(),
        mesh,
        IOobject::NO_READ,
        IOobject::AUTO_WRITE
    ),
    4*sin(2*3.14* strainMag/S0)
    + 6*sqr(strainMag/S0)
    + 5*exp(strainMag/S0)
	
);

PrSNonlinear.write();

volScalarField PrSW
(
    IOobject
    (
        "PrSW",
        runTime.timeName(),
        mesh,
        IOobject::NO_READ,
        IOobject::AUTO_WRITE
    ),
	( 4*sin(2*3.14* strainMag/S0)
	+ 6*sqr(strainMag/S0)
	+ 5*exp(strainMag/S0) 
	)  / (scalar(1) + sqr(omegaMag/S0))	

);

PrSW.write();

volScalarField Pr(PrSLinear * gammaSLinear + PrSNonlinear * gammaSNonlinear);
*/


#include "createPhi.H"
#include "createFvOptions.H"
