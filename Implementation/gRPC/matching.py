#from ordered_set import OrderedSet
import numpy as np
import math


class Matching():
    def __init__(self,):
        #self.U = ['A','B','C']
        self.U = ['192.168.0.105']
        #self.F = [1,2,3,4,5]
        self.F = ['192.168.0.106','192.168.0.107']
        self.U_preference_list = {}
        self.F_preference_list = {}
        self.U_utilities = {}
        self.F_utilities = {}
        self.U_match = {}
        self.F_match = {}
        self.U_CA = []
        self.F_CA = []
        self.rand_match = []
        self.freeman = None
        self.calculate_utilities()
        self.initilize_match()
        self.random_match()
        
        
    def print_(self,title):
        if title == "U_F_utilities":
            print("U_utilities:")
            for u in self.U_utilities:
                print("{}: {}".format(u,self.U_utilities[u]))
                
            print("F_utilities:")
            for f in self.F_utilities:
                print("{}: {}".format(f,self.F_utilities[f]))
        
        elif title == "U_F_preference_list":
            print("U_preference_list:")
            for u in self.U_preference_list:
                print("{}: {}".format(u,self.U_preference_list[u]))
            
            print("F_preference_list:")
            for f in self.F_preference_list:
                print("{}: {}".format(f,self.F_preference_list[f]))
                
    def calculate_utilities(self):
        # Node: calculate utility of all fog nodes for each User 
        if len(self.F) > 1:
            for i in self.U:
                u_l = {}
                for j in self.F:
                    temp = float(np.random.randint(1, 10))
                    u_l.update({j:temp})
                    self.F_CA.append(temp)
                self.U_utilities.update({i:u_l})
        else:
            j = self.F
            u_l = {}
            temp = float(np.random.randint(1, 10))
            u_l.update({j[0]:temp})
            self.F_CA.append(temp)
            for i in self.U:
                self.U_utilities.update({i:u_l})
    
    
        for u in self.U_utilities:
            l = []
            f_u = self.U_utilities[u]
            f_u = {k: v for k, v in sorted(f_u.items(), key=lambda item: item[1], reverse=True)}
            for k in f_u:
                l.append(k)
            self.U_preference_list.update({u:l})

        
        
        # Node: calculate utility of all users for each fog node
        if len(self.U) > 1:
            for i in self.F:
                u_l = {}
                for j in self.U:
                    temp = float(np.random.randint(1, 10))
                    u_l.update({j:temp})
                    self.U_CA.append(temp)
                self.F_utilities.update({i:u_l})
        else:
            j = self.U
            u_l = {}
            temp = float(np.random.randint(1, 10))
            u_l.update({j[0]:temp})
            self.U_CA.append(temp)
            for i in self.F:
                self.F_utilities.update({i:u_l})
                
        for f in self.F_utilities:
            l = []
            f_u = self.F_utilities[f]
            f_u = {k: v for k, v in sorted(f_u.items(), key=lambda item: item[1], reverse=True)}
            for k in f_u:
                l.append(k)
            self.F_preference_list.update({f:l})

        
        # print
        self.print_("U_F_utilities")
        self.print_("U_F_preference_list")

    """
    def random_match(self):
        nodes = [x for x in self.F]
        for u in self.U:
            n_l = len(nodes)
            if n_l == 0:
                nodes = [x for x in self.F]
            index = np.random.randint(0,len(nodes))
            f = nodes.pop(index)
            self.rand_match.append((u,f))
    """
    """
    def random_match(self):
        for u in self.U:
            l = len(self.U_preference_list[u])
            index = np.random.randint(0,math.ceil(l/2))
            f = self.U_preference_list[u][index]
            self.rand_match.append((u,f))
        print(self.rand_match)
    """
    
    
    def random_match(self):
        for u in self.U:
            idx = np.random.randint(0,len(self.F))
            f = self.F[idx]
            self.rand_match.append((u,f))

    def greedy_random_match(self):
        random_match = [x for x in self.rand_match]
        return random_match
    
    
           
    
            
    # initializing all agent to free
    def initilize_match(self):
        for u in self.U:
            self.U_match.update({u:0.0})
        for f in self.F:
            self.F_match.update({f:0.0})
    """     
    def DNN_inference_offloading_swap_matching(self):
        #print("Random Match: {}".format(self.rand_match))
        #l = len(self.rand_match)
        iterator = 0
        print("Random Match: {}".format(self.rand_match))
        l = len(self.rand_match)
        for pair in range(l-1):
            for swap_pair in range(l):
                if pair != swap_pair:
                    will_swap = self.check_preference(pair,swap_pair)
                    if will_swap:
                        u = self.rand_match[pair][0]
                        u_ = self.rand_match[swap_pair][0]
                        f = self.rand_match[pair][1]
                        f_ = self.rand_match[swap_pair][1]
                        ufutility = self.U_utilities[u][f] + self.F_utilities[f][u]
                        u_f_untility = self.U_utilities[u_][f_] + self.F_utilities[f_][u_]
                        new_pair_1 = (u,f_)
                        new_pair_2 = (u_,f)
                        uf_utility = self.U_utilities[u][f_] + self.F_utilities[f_][u]
                        u_funtility = self.U_utilities[u_][f] + self.F_utilities[f][u_]
                        self.rand_match.pop(pair)
                        self.rand_match.pop(swap_pair-1)
                        self.rand_match.insert(pair,new_pair_1)
                        self.rand_match.insert(swap_pair,new_pair_2)
                        print("Swap: pair:{} swap_pair:{} - {}".format(pair,swap_pair,self.rand_match))
                        print("Utility (Before): uf {} - u_f_ {}".format(ufutility,u_f_untility))
                        print("Utility (After): uf_ {} - u_f {}".format(uf_utility,u_funtility))
        print("Match (After swapping): " + str(self.rand_match))
        iterator += 1
        """
    def DNN_inference_offloading_swap_matching(self):
        #print("Random Match: {}".format(self.rand_match))
        #l = len(self.rand_match)
        iterator = 0
        print("Random Match: {}".format(self.rand_match))

        l = len(self.rand_match)
        if l > 1:
            for pair in range(l-1):
                for swap_pair in range(l):
                    if pair != swap_pair:
                        will_swap = self.check_preference(pair,swap_pair)
                        if will_swap:
                            u = self.rand_match[pair][0]
                            u_ = self.rand_match[swap_pair][0]
                            f = self.rand_match[pair][1]
                            f_ = self.rand_match[swap_pair][1]
                            ufutility = self.U_utilities[u][f] + self.F_utilities[f][u]
                            u_f_untility = self.U_utilities[u_][f_] + self.F_utilities[f_][u_]
                            new_pair_1 = (u,f_)
                            new_pair_2 = (u_,f)
                            uf_utility = self.U_utilities[u][f_] + self.F_utilities[f_][u]
                            u_funtility = self.U_utilities[u_][f] + self.F_utilities[f][u_]
                            self.rand_match.pop(pair)
                            self.rand_match.pop(swap_pair-1)
                            self.rand_match.insert(pair,new_pair_1)
                            self.rand_match.insert(swap_pair,new_pair_2)
                            print("Swap: pair:{} swap_pair:{} - {}".format(pair,swap_pair,self.rand_match))
                            print("Utility (Before): uf {} - u_f_ {}".format(ufutility,u_f_untility))
                            print("Utility (After): uf_ {} - u_f {}".format(uf_utility,u_funtility))
            print("Match (After swapping): " + str(self.rand_match))
            #return self.rand_match
        else:
            U = self.U[0]
            U_prefer = self.U_preference_list[U][0]
            f_r_match = self.rand_match[0][1]
            if f_r_match != U_prefer:
                new_pair = (U,U_prefer)
                self.rand_match.pop(0)
                self.rand_match.insert(0,new_pair)
            print("Match (After swapping): " + str(self.rand_match))
        
        iterator += 1

    def check_preference(self,p,s_p):
        u = self.rand_match[p][0]
        u_ = self.rand_match[s_p][0]
        f = self.rand_match[p][1]
        f_ = self.rand_match[s_p][1]
        if self.preference(self.U_preference_list,u,f,f_):
            if self.preference(self.U_preference_list, u_,f_,f):
                if self.preference(self.F_preference_list,f,u,u_):
                    if self.preference(self.F_preference_list,f_,u_,u):
                        return True
        return False
    
    def preference(self,P,agent,partner,candidate):
        prefer_list = P[agent]
        p_index = prefer_list.index(partner)
        c_index = prefer_list.index(candidate)
        if p_index >= c_index:
            return True
        else:
            return False
        

