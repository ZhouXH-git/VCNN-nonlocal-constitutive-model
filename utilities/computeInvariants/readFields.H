    volSymmTensorField R
    (
        IOobject
        (
            "R2",
            runTime.timeName(),
            mesh,
            IOobject::MUST_READ
        ),
        mesh
    );
    
volScalarField A2
    (
        IOobject
        (
            "A2",
            runTime.timeName(),
            mesh,
            IOobject::NO_READ,
            IOobject::AUTO_WRITE
        ),
        mesh,
        dimensionedScalar("A2v", dimless, 0.0)
    );

    volScalarField xi
    (
        IOobject
        (
            "xi",
            runTime.timeName(),
            mesh,
            IOobject::NO_READ,
            IOobject::AUTO_WRITE
        ),
        mesh,
        dimensionedScalar("xiv", dimless, 0.0)
    );

    volScalarField eta
    (
        IOobject
        (
            "eta",
            runTime.timeName(),
            mesh,
            IOobject::NO_READ,
            IOobject::AUTO_WRITE
        ),
        mesh,
        dimensionedScalar("etav", dimless, 0.0)
    );


    volSymmTensorField b
    (
        IOobject
        (
            "b",
            runTime.timeName(),
            mesh,
            IOobject::NO_READ,
            IOobject::AUTO_WRITE
        ),
        mesh,
        dimensionedSymmTensor("bv", dimless, symmTensor::zero)
    );

    volVectorField lam
    (
        IOobject
        (
            "lam",
            runTime.timeName(),
            mesh,
            IOobject::NO_READ,
            IOobject::AUTO_WRITE
        ),
        mesh,
        dimensionedVector ("lamv", dimless, vector::zero)
    );


