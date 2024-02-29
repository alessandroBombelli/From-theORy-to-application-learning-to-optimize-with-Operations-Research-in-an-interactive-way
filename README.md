# From theORy to application: learning to optimize with Operations Research in an interactive way

This repository contains the main outcomes of an Open Education Stimulation Fund Project (https://www.tudelft.nl/en/open-science/funding/awarded-projects/open-education-stimulation-fund-2022) funded by the Tu Delft Library

This repository contains folly worked-out examples of several Operations Research (OR) problems, such as shortest path, vehicle routing, bin-packing problems. 
Each problem is briefly introduced and the basics of the underlying mathematical formulation are provided. Then, such formulation is translated into a 
Python code that sets up the problem and solves it using either the commercial solver Gurobi (https://www.gurobi.com/. To use Gurobi, a license is needed, which is 
free (academic version) for students and all interested users affiliated to a university) or an open-source solver via implementation with PuLP (https://coin-or.github.io/pulp/). While the first option is always given due to the expertise and preferred coding routing of the authors, the second option is provided when possible to enhance the open-source dissemination of the repository.

The target audience is people with limited knowledge of OR. Hence, we try to engage users with interesting examples and relate them to practical problems
to show the potential (or at least the tip of the iceberg) of OR modeling. All OR problems are solved with Branch and Bound with default settings, 
unless differently specified. This repository is about translating OR problems into OR models and, as such, does not contain algorithmic-oriented examples
such as column generation, matheuristic, branch-and-price, etc.

When possible, we will pair OR problems with a serious game equivalent (e.g., a print-and-play board game) to enhance engagement and interaction among users. As this is a constantly-updated work in progress, we are happy to receive any feedback concerning typos, errors, models or material that users would like to be uploaded. You can reach out to me (Alessandro Bombelli) at a.bombelli@tudelft.nl.

Other contributors are:
 - Bilge Atasoy (b.atasoy@tudeflt.nl)
 - Doris Boschma (d.boschma@tudelft.nl)
 - Stefano Fazi (s.fazi@tudelft.nl)
