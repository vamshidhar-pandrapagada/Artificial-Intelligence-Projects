from aimacode.logic import PropKB
from aimacode.planning import Action
from aimacode.search import (
    Node, Problem,
)
from aimacode.utils import expr
from lp_utils import (
    FluentState, encode_state, decode_state,
)
from my_planning_graph import PlanningGraph

from functools import lru_cache


class AirCargoProblem(Problem):
    def __init__(self, cargos, planes, airports, initial: FluentState, goal: list):
        """

        :param cargos: list of str
            cargos in the problem
        :param planes: list of str
            planes in the problem
        :param airports: list of str
            airports in the problem
        :param initial: FluentState object
            positive and negative literal fluents (as expr) describing initial state
        :param goal: list of expr
            literal fluents required for goal test
        """
        self.state_map = initial.pos + initial.neg
        self.initial_state_TF = encode_state(initial, self.state_map)
        Problem.__init__(self, self.initial_state_TF, goal=goal)
        self.cargos = cargos
        self.planes = planes
        self.airports = airports
        self.actions_list = self.get_actions()

    def get_actions(self):
        """
        This method creates concrete actions (no variables) for all actions in the problem
        domain action schema and turns them into complete Action objects as defined in the
        aimacode.planning module. It is computationally expensive to call this method directly;
        however, it is called in the constructor and the results cached in the `actions_list` property.

        Returns:
        ----------
        list<Action>
            list of Action objects
        """

        # TODO create concrete Action objects based on the domain action schema for: Load, Unload, and Fly
        # concrete actions definition: specific literal action that does not include variables as with the schema
        # for example, the action schema 'Load(c, p, a)' can represent the concrete actions 'Load(C1, P1, SFO)'
        # or 'Load(C2, P2, JFK)'.  The actions for the planning problem must be concrete because the problems in
        # forward search and Planning Graphs must use Propositional Logic

        def load_actions():
            """Create all concrete Load actions and return a list

            :return: list of Action objects
            """
            loads = []
            for air in self.airports:
                for p in self.planes:
                    for c in self.cargos:
                            precond_cargo_pos = [expr("At({}, {})".format(c, air)),expr("At({}, {})".format(p, air))]
                            precond_cargo_neg = []
                            effect_cargo_add = [expr("In({}, {})".format(c, p))]
                            effect_cargo_rem = [expr("At({}, {})".format(c, air))]
                            load = Action(expr("Load({}, {}, {})".format(c,p,air)),
                                            [precond_cargo_pos, precond_cargo_neg],
                                            [effect_cargo_add, effect_cargo_rem]
                                            )
                            loads.append(load)
            
            # TODO create all load ground actions from the domain Load action
            return loads

        def unload_actions():
            """Create all concrete Unload actions and return a list

            :return: list of Action objects
            """
            unloads = []
            for air in self.airports:
                for p in self.planes:
                    for c in self.cargos:
                            precond_cargo_pos = [expr("In({}, {})".format(c, p)),expr("At({}, {})".format(p, air))]
                            precond_cargo_neg = []
                            effect_cargo_add = [expr("At({}, {})".format(c, air))]
                            effect_cargo_rem = [expr("In({}, {})".format(c, p))]
                            unload = Action(expr("Unload({}, {}, {})".format(c,p,air)),
                                            [precond_cargo_pos, precond_cargo_neg],
                                            [effect_cargo_add, effect_cargo_rem]
                                            )
                            unloads.append(unload)
            # TODO create all Unload ground actions from the domain Unload action
            return unloads

        def fly_actions():
            """Create all concrete Fly actions and return a list

            :return: list of Action objects
            """
            flys = []
            for fr in self.airports:
                for to in self.airports:
                    if fr != to:
                        for p in self.planes:
                            precond_pos = [expr("At({}, {})".format(p, fr)),
                                           ]
                            precond_neg = []
                            effect_add = [expr("At({}, {})".format(p, to))]
                            effect_rem = [expr("At({}, {})".format(p, fr))]
                            fly = Action(expr("Fly({}, {}, {})".format(p, fr, to)),
                                         [precond_pos, precond_neg],
                                         [effect_add, effect_rem])
                            flys.append(fly)
            return flys

        return load_actions() + unload_actions() + fly_actions()

    def actions(self, state: str) -> list:
        """ Return the actions that can be executed in the given state.

        :param state: str
            state represented as T/F string of mapped fluents (state variables)
            e.g. 'FTTTFF'
        :return: list of Action objects
        """
        # TODO implement
        possible_actions = []
        kb = PropKB()
        pos_sentence = decode_state(state,self.state_map).pos_sentence()
        kb.tell(pos_sentence)
        
        for actions in self.actions_list:
            postives = all([clause in kb.clauses for clause in actions.precond_pos])
            negatives = all([clause not in kb.clauses for clause in actions.precond_neg])
            if postives and negatives: 
                possible_actions.append(actions)       
        
        return possible_actions     
        
              
    
    def result(self, state: str, action: Action):
        """ Return the state that results from executing the given
        action in the given state. The action must be one of
        self.actions(state).

        :param state: state entering node
        :param action: Action applied
        :return: resulting state after action
        """
        # TODO implement
        new_state = FluentState([], [])
        previous_state = decode_state(state,self.state_map)
        
        # Carry over the result of positive effects to the new state
        new_state.pos.extend(pred for pred in previous_state.pos if pred not in action.effect_rem)
        new_state.pos.extend(pred for pred in action.effect_add if pred not in new_state.pos)
        
        # Carry over the result of negative effects to the new state
        new_state.neg.extend(pred for pred in previous_state.neg if pred not in action.effect_add)
        new_state.neg.extend(pred for pred in action.effect_rem if pred not in new_state.neg)      
        
        
        return encode_state(new_state, self.state_map)

    def goal_test(self, state: str) -> bool:
        """ Test the state to see if goal is reached

        :param state: str representing state
        :return: bool
        """
        kb = PropKB()
        kb.tell(decode_state(state, self.state_map).pos_sentence())
        for clause in self.goal:
            if clause not in kb.clauses:
                return False
        return True

    def h_1(self, node: Node):
        # note that this is not a true heuristic
        h_const = 1
        return h_const

    @lru_cache(maxsize=8192)
    def h_pg_levelsum(self, node: Node):
        """This heuristic uses a planning graph representation of the problem
        state space to estimate the sum of all actions that must be carried
        out from the current state in order to satisfy each individual goal
        condition.
        """
        # requires implemented PlanningGraph class
        pg = PlanningGraph(self, node.state)
        pg_levelsum = pg.h_levelsum()
        return pg_levelsum

    @lru_cache(maxsize=8192)
    def h_ignore_preconditions(self, node: Node):
        """This heuristic estimates the minimum number of actions that must be
        carried out from the current state in order to satisfy all of the goal
        conditions by ignoring the preconditions required for an action to be
        executed.
        """
        count = 0
        
        kb = PropKB()
        
        #Get all the Positive Preconditions from the knowledge base created by propositional logic
        kb.tell(decode_state(node.state, self.state_map).pos_sentence())
        
        # if precondition clauses not in goal, increase the heuristic value
        for clause in self.goal:
            if clause not in kb.clauses:
                count += 1
        return count    
       


def air_cargo_p1() -> AirCargoProblem:
    cargos = ['C1', 'C2']
    planes = ['P1', 'P2']
    airports = ['SFO', 'JFK']
    pos = [expr('At(C1, SFO)'),
           expr('At(C2, JFK)'),
           expr('At(P1, SFO)'),
           expr('At(P2, JFK)'),
           ]
    neg = [expr('At(C2, SFO)'),
           expr('In(C2, P1)'),
           expr('In(C2, P2)'),
           expr('At(C1, JFK)'),
           expr('In(C1, P1)'),
           expr('In(C1, P2)'),
           expr('At(P1, JFK)'),
           expr('At(P2, SFO)'),
           ]
    init = FluentState(pos, neg)
    goal = [expr('At(C1, JFK)'),
            expr('At(C2, SFO)'),
            ]
    return AirCargoProblem(cargos, planes, airports, init, goal)


def air_cargo_p2() -> AirCargoProblem:
    # TODO implement Problem 2 definition
    cargos = ['C1', 'C2','C3']
    planes = ['P1', 'P2','P3']
    airports = ['SFO', 'JFK','ATL']
    
    pos = [expr('At(C1, SFO)'),
           expr('At(C2, JFK)'),
           expr('At(C3, ATL)'),
           expr('At(P1, SFO)'),
           expr('At(P2, JFK)'),
           expr('At(P3, ATL)'),
           ]
    neg = []
    for c in cargos:
        for p in planes:
            neg.append(expr("In({}, {})".format(c, p)))
    cargos_planes = cargos + planes
    temp_list = []
    for air in airports:
        if (air =='SFO'):
            temp_list.append([expr("At({}, {})".format(val, air)) for val in cargos_planes if val not in ('C1','P1')])
        elif (air =='JFK'):
            temp_list.append([expr("At({}, {})".format(val, air)) for val in cargos_planes if val not in ('C2','P2')])
        elif (air =='ATL'):
            temp_list.append([expr("At({}, {})".format(val, air)) for val in cargos_planes if val not in ('C3','P3')])
        
    neg = neg + [item for items in temp_list for item in items] 
   
    init = FluentState(pos, neg)
    goal = [expr('At(C1, JFK)'),
            expr('At(C2, SFO)'),
            expr('At(C3, SFO)'),
            ]
    
    return AirCargoProblem(cargos, planes, airports, init, goal)


def air_cargo_p3() -> AirCargoProblem:
    # TODO implement Problem 3 definition
    cargos = ['C1', 'C2','C3','C4']
    planes = ['P1', 'P2']
    airports = ['SFO', 'JFK','ATL','ORD']
    
    pos = [expr('At(C1, SFO)'),
           expr('At(C2, JFK)'),
           expr('At(C3, ATL)'),
           expr('At(C4, ORD)'),
           expr('At(P1, SFO)'),
           expr('At(P2, JFK)'),           
           ]
    
    neg = []
    for c in cargos:
        for p in planes:
            neg.append(expr("In({}, {})".format(c, p)))
    cargos_planes = cargos + planes
    temp_list = []
    for air in airports:
        if (air =='SFO'):
            temp_list.append([expr("At({}, {})".format(val, air)) for val in cargos_planes if val not in ('C1','P1')])
        elif (air =='JFK'):
            temp_list.append([expr("At({}, {})".format(val, air)) for val in cargos_planes if val not in ('C2','P2')])
        elif (air =='ATL'):
            temp_list.append([expr("At({}, {})".format(val, air)) for val in cargos_planes if val not in ('C3')])
        elif (air =='ORD'):
            temp_list.append([expr("At({}, {})".format(val, air)) for val in cargos_planes if val not in ('C4')])
    
    neg = neg + [item for items in temp_list for item in items] 
   
    init = FluentState(pos, neg)
    
    goal = [expr('At(C1, JFK)'),
            expr('At(C3, JFK)'),
            expr('At(C2, SFO)'),
            expr('At(C4, SFO)'),
            ]
    
    return AirCargoProblem(cargos, planes, airports, init, goal)
    