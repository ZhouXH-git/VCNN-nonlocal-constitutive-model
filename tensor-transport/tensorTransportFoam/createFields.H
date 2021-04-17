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

// Velocity field is frozen. Compute gradU outside the loop
tmp<volTensorField> tgradU(fvc::grad(U));
const volTensorField& gradU = tgradU();


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
//  (
//      dimensioned<scalar>::lookupOrAddToDict
//      (
//          "nu",
//          transportProperties,
//          1E-6
//      )
//  );

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

transportProperties.lookupOrAddDefault<Switch>("initRwithChannel", false);

// mixing length: lm = min(κη, Cμδ)

// TODO: copy mixing length computation here:
// need to set up κ  and wall distance κη as well; Cmu is constructed above.

// used only if using effective viscsity to diffuse R
// volScalarField DREff
//  (
//      IOobject
//      (
//          "DREff",
//          runTime.timeName(),
//          mesh,
//          IOobject::NO_READ,
//          IOobject::AUTO_WRITE
//      ),
//      Cc * sqr(k) * lm + nu
//  );

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


if(initRwithChannel)
{
    dimensionedScalar uprime // turb. intensity to initialize R
        (
            transportProperties.lookup("uprime")
        );

    volScalarField nut(Cc * uprime * lm);
    volSymmTensorField twoS(twoSymm(gradU));
    volScalarField shear = nut*twoS.component(symmTensor::XY);

    // Follow Pope (2000), Fig. 7.17/7.33 for ratio of stress components
    // <u^2>:<v^2>:<w^2>:<uv> = 1:0.4:0.6:(-0.5)
    Info << "Replacing R with plane channel flow ratio ..." <<endl;
    R.replace(tensor::XY, shear);
    R.replace(tensor::XX, -2*shear);
    R.replace(tensor::YY, -0.8*shear);
    R.replace(tensor::ZZ, -1.2*shear);
    R.replace(tensor::XZ, 0.0);
    R.replace(tensor::YZ, 0.0);

    volSymmTensorField RChannel
        (
            "RChannel",
            R
        );
    RChannel.write();
}




#include "createPhi.H"

#include "createFvOptions.H"
