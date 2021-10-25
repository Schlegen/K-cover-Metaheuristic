export CPLEX_HOME="/opt/ibm/ILOG/CPLEX_Studio128/cplex"
export CPO_HOME="/opt/ibm/ILOG/CPLEX_Studio128/cpoptimizer"
export PATH="${PATH}:${CPLEX_HOME}/bin/x86-64_linux:${CPO_HOME}/bin/x86-64_linux"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${CPLEX_HOME}/bin/x86-64_linux:${CPO_HOME}/bin/x86-64_linux"
export PYTHONPATH="${PYTHONPATH}:/opt/ibm/ILOG/CPLEX_Studio128/cplex/python/3.5/x86-64_linux"

python3 main.py