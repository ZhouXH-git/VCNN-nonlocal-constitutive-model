volScalarField k = 0.5*tr(R);
dimensionedScalar kSmall("ks",dimVelocity*dimVelocity,SMALL);
// b = R/(2.0*k+kSmall) - 1.0/3.0*I;
b = dev(R)/(2.0*k+kSmall);
lams = eigenValues(b);

// reverse order, since lambdas given by OF is in increasing order
// const volScalarField& L1 = lams.component(vector::Z);
// const volScalarField& L2 = lams.component(vector::Y);
// const volScalarField& L3 = lams.component(vector::X);


// Lumley triangle
A2 = sqr(lams.component(vector::Z)) + sqr(lams.component(vector::Y))
    + lams.component(vector::Z) * lams.component(vector::Y);
A3 =  -lams.component(vector::Z) * lams.component(vector::Y)
    * (lams.component(vector::Z) + lams.component(vector::Y));

// Compute Barycentric coordinates
volScalarField C1 = lams.component(vector::Z) - lams.component(vector::Y); //L1 - L2
volScalarField C2 = 2.0*(lams.component(vector::Y) - lams.component(vector::X)); //2.0*(L2 - L3)
volScalarField C3 =  3.0*lams.component(vector::X) + 1.0; //3.0*L3 + 1.0;
 
scalarList xc(3, 0.0);
scalarList yc(3, 0.0);
xc[0] = 1.0; xc[1] = 0.0; xc[2] = 0.5;
yc[0] = 0.0; yc[1] = 0.0; yc[2] = Foam::sqrt(3.0)/2;

// Generic from
//xB = C1 * xc[0] + C2 * xc[1] + C3 * xc[2];
//yB = C1 * yc[0] + C2 * yc[1] + C3 * yc[2]; 

// Dropping zero terms simplies to (Emory 2014; Eq. 2.3)
xB = C1 * xc[0] + C3 * xc[2];
yB = C3 * yc[2];

b.write();
lams.write();
A2.write(); A3.write();

xB.write(); yB.write();

