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


Info<< "Reading field R\n" << endl;

volSymmTensorField R
(
    IOobject
    (
        "R",
        runTime.timeName(),
        mesh,
        IOobject::MUST_READ,
        IOobject::AUTO_WRITE
    ),
    mesh
);

volSymmTensorField Pr
(
    IOobject
    (
        "Pr",
        runTime.timeName(),
        mesh,
        IOobject::NO_READ,
        IOobject::AUTO_WRITE
    ),
    -twoSymm(R & ((fvc::grad(U))()) )
);

volScalarField k
(
    IOobject
    (
        "k",
        runTime.timeName(),
        mesh,
        IOobject::NO_READ,
        IOobject::AUTO_WRITE
    ),
    0.5*tr(R)
);
       


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


Info<< "Reading diffusivity and coefficients\n" << endl;

dimensionedScalar DR
(
    transportProperties.lookup("DR")
);

// used only if using effective viscsity to diffuse R
// dimensionedScalar nu
// (
//     dimensioned<scalar>::lookupOrAddToDict
//     (
//         "nu",
//         transportProperties,
//         1E-8
//     )
// );

dimensionedScalar C1
(
    dimensioned<scalar>::lookupOrAddToDict
    (
        "C1",
        transportProperties,
        1.8
    )
);

dimensionedScalar C2
(
    dimensioned<scalar>::lookupOrAddToDict
    (
        "C2",
        transportProperties,
        0.6
    )
);

dimensionedScalar Cmu
(
    dimensioned<scalar>::lookupOrAddToDict
    (
        "Cmu",
        transportProperties,
        0.09
    )
);

// epsilon = Cd k^(3/2) / lm
// nu_t = Cmu k^2/epsilon
// Cmu = c^4; Cd = c^3 --> Cc = Cmu^(1/4)
// see (Pope 2000) pp. 373; Eq. (10.44)-(10.47)
dimensionedScalar Cc(sqrt(sqrt(Cmu)));
dimensionedScalar Cd(pow(Cc, 3));



// mixing length: lm = min(κη, Cμδ)

// TODO: copy mixing length computation here:
// need to set up κ  and wall distance κη as well; Cmu is constructed above.

// used only if using effective viscsity to diffuse R
// volScalarField DREff
// (
//     IOobject
//     (
//         "DREff",
//         runTime.timeName(),
//         mesh,
//         IOobject::NO_READ,
//         IOobject::AUTO_WRITE
//     ),
//     Cc * sqr(k) * lm + nu)
// );

volScalarField epsilon
(
    IOobject
    (
        "epsilon",
        runTime.timeName(),
        mesh,
        IOobject::NO_READ,
        IOobject::AUTO_WRITE
    ),
    Cd*pow(sqrt(mag(k)), 3)/lm
);


#include "createPhi.H"

#include "createFvOptions.H"
