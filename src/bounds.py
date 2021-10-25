from os import times
from solution_class import Solution
import pulp as pl
import networkx as nx
from utils.math_utils import dist
import cplex


class MilpApproach(Solution):
    def __init__(self, instance):
        flow_value = instance.k * instance.n_targets
        self.graph = nx.DiGraph()
        self.graph.add_nodes_from([(0, e[0], e[1]) for e in instance.targets])
        self.graph.add_nodes_from([(1, e[0], e[1]) for e in instance.targets])
        self.graph.add_node((1, 0, 0))
        self.graph.add_node((2, 0, 0))
        E_capt = instance.neighbours(instance.Rcapt, take_origin=False)
        self.graph.add_edges_from([((0, u[0], u[1]), (1, v[0], v[1])) for u, v in E_capt], capacity=1)
        self.graph.add_edges_from([((0, u[0], u[1]), (1, u[0], u[1])) for u in instance.targets])
        E_com = instance.neighbours(instance.Rcom, take_origin=True)
        self.graph.add_edges_from([((1, u[0], u[1]), (1, v[0], v[1])) for u, v in E_com])
        self.graph.add_edges_from([((1, v[0], v[1]), (2, 0, 0)) for v in instance.targets if dist((0,0), v) <= instance.Rcom])

        self._compute_relaxation(instance)
        self._compute_milp(instance)
        #print(model)

        self.captors = []
    
    def _compute_milp(self, instance):
        model = pl.LpProblem("model", pl.LpMinimize)
        
        #variables
        self.flow_vars = {e : pl.LpVariable("f_"+str(e), lowBound = 0, cat="Continuous") for e in self.graph.edges}
        self.x_vars = {n : pl.LpVariable("x_"+str(n), cat="Binary") for n in self.graph.nodes if n[0] == 1}

        #objectif
        model += pl.lpSum([self.x_vars[n] for n in self.graph.nodes if n[0] == 1]), "Nombre de capteurs"
        
        #contraintes
        for n in self.graph.nodes:
            if n[0] == 0 :
                model += pl.lpSum([self.flow_vars[e] for e in self.graph.edges if e[0] == n]) == instance.k

            elif n[0] == 1 :
                model += pl.lpSum([self.flow_vars[e] for e in self.graph.edges if e[0] == n]) - \
                     pl.lpSum([self.flow_vars[e] for e in self.graph.edges if e[1] == n]) == 0

                list_inflow_from_0 = [self.flow_vars[e] for e in self.graph.edges if (e[1] == n and e[0][0] ==0)]
                model += len(list_inflow_from_0) * self.x_vars[n] - pl.lpSum(list_inflow_from_0) >= 0
                
                model += instance.k * instance.n_targets * self.x_vars[n] - pl.lpSum([self.flow_vars[e] for e in self.graph.edges if e[1] == n]) >= 0

                # tester avec ou sans - > traiter comme une coupe
                if n[1] != 0 or n[2] != 0:
                    model += self.x_vars[n] - self.flow_vars[((0, n[1], n[2]), n)] == 0

            elif n[0] == 2 :
                model += pl.lpSum([self.flow_vars[e] for e in self.graph.edges if e[1] == n]) == instance.k * instance.n_targets

        for e in self.graph.edges:
            if e[0][0] == 0:
                model += self.flow_vars[e] <= 1

        self.model_milp = model

    def _compute_relaxation(self, instance):
        model = pl.LpProblem("model", pl.LpMinimize)
        
        #variables
        self.flow_vars = {e : pl.LpVariable("f_"+str(e), lowBound = 0, cat="Continuous") for e in self.graph.edges}
        self.x_vars = {n : pl.LpVariable("x_"+str(n), lowBound = 0, upBound=1, cat="Continuous") for n in self.graph.nodes if n[0] == 1}

        #objectif
        model += pl.lpSum([self.x_vars[n] for n in self.graph.nodes if n[0] == 1]), "Nombre de capteurs"
        
        #contraintes
        for n in self.graph.nodes:
            if n[0] == 0 :
                model += pl.lpSum([self.flow_vars[e] for e in self.graph.edges if e[0] == n]) == instance.k

            elif n[0] == 1 :
                model += pl.lpSum([self.flow_vars[e] for e in self.graph.edges if e[0] == n]) - \
                     pl.lpSum([self.flow_vars[e] for e in self.graph.edges if e[1] == n]) == 0

                list_inflow_from_0 = [self.flow_vars[e] for e in self.graph.edges if (e[1] == n and e[0][0] ==0)]
                model += len(list_inflow_from_0) * self.x_vars[n] - pl.lpSum(list_inflow_from_0) >= 0
                
                model += instance.k * instance.n_targets * self.x_vars[n] - pl.lpSum([self.flow_vars[e] for e in self.graph.edges if e[1] == n]) >= 0

                # tester avec ou sans - > traiter comme une coupe
                if n[1] != 0 or n[2] != 0:
                    model += self.x_vars[n] - self.flow_vars[((0, n[1], n[2]), n)] == 0

            elif n[0] == 2 :
                model += pl.lpSum([self.flow_vars[e] for e in self.graph.edges if e[1] == n]) == instance.k * instance.n_targets

        for e in self.graph.edges:
            if e[0][0] == 0:
                model += self.flow_vars[e] <= 1

        self.model_relaxation = model

    def exact_solution(self, time_limit=60):
        solver = pl.CPLEX_PY()
        solver.buildSolverModel(self.model_milp)

        #Modify the solver model
        if time_limit:
            solver.solverModel.parameters.timelimit.set(time_limit)

        solver.callSolver(self.model_milp)
        status = solver.findSolutionValues(self.model_milp)
        self.exact_solution_value = pl.value(self.model_milp.objective)
        #print((e[1], e[2]) for e in self.x_vars.keys())
        self.captors = [(e[1], e[2]) for e in self.x_vars.keys() if self.x_vars[e].value() > 0.5]
        print(self.captors)

    def relaxation_value(self, time_limit=60):
        solver = pl.CPLEX_PY()
        solver.buildSolverModel(self.model_relaxation)

        #Modify the solver model
        if time_limit:
            solver.solverModel.parameters.timelimit.set(time_limit)

        solver.callSolver(self.model_relaxation)
        status = solver.findSolutionValues(self.model_relaxation)
        self.relaxation_value = pl.value(self.model_relaxation.objective)

    