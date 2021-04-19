volScalarField k = 0.5*tr(R);
dimensionedScalar kSmall("ks",dimVelocity*dimVelocity,SMALL);
b = R/(2.0*k+kSmall) - 1.0/3.0*I;
lam = eigenValues(b);
A2 = tr(b & b);
// A3 =  det(b);
//A2.write();
b.write();
lam.write();

volScalarField C1 = lam.component(vector::Z) - lam.component(vector::Y);
volScalarField C2 = 2.0*(lam.component(vector::Y) - lam.component(vector::X));
volScalarField C3 = 3.0*lam.component(vector::X) + 1.0;
 
scalarList xc(3, 0.0);
scalarList yc(3, 0.0);
xc[0] = 1.0; xc[1] = 0.0; xc[2] = 0.5;
yc[0] = 0.0; yc[1] = 0.0; yc[2] = Foam::sqrt(3.0)/2;


xi = C1 * xc[0] + C2 * xc[1] + C3 * xc[2];
eta = C1 * yc[0] + C2 * yc[1] + C3 * yc[2]; 
xi.write();
eta.write();
