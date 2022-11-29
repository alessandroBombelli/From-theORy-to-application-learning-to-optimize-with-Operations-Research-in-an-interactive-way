# from_theORy_to_application

This repository contains folly worked-out examples of several Operations Research (OR) problems, such as shortet path, vehicle routing, bin-packing problems. 
Each problem is briefly introduced and the basics of the underlying mathematical formulation are provided. Then, such formulation is translated into a 
Python code that sets up the problem and solves it using the commercial solver Gurobi (https://www.gurobi.com/). To use Gurobi, a license is needed, which is 
free (academic version) for students of people affiliated to a university.

The target audience is people with limited knowledge of OR. Hence, we try to engage users with interesting examples and to relate them to practical problems
to show the potential (or at least the tipof the iceberg) of OR modeling. All OR problems are solved with Branch and Bound with default settings, 
unless differently speficied. This repository is about translating OR problems into OR models and, as such, does not contain algorithmic-oriented examples
such as column generation, matheuristic, branch-and-price, etc.

When possible, we will pair OR problems with a serious game equivament (e.g., a print-and-play board game) to enhance engagement and interaction among users.

Other contributors are:
 - Bilge Atasoy (b.atasoy@tudeflt.nl)
 - Doris Boschma (d.boschma@tudelft.nl)
 - Stefano Fazi (s.fazi@tudelft.nl)
