import numpy as np

np.random.seed(42)

class WhaleOptimization():
    """class implements the whale optimization algorithm as found at
    http://www.alimirjalili.com/WOA.html
    and
    https://doi.org/10.1016/j.advengsoft.2016.01.008
    """
    def __init__(self, opt_func, constraints, nsols, b, a, a_step, maximize=False):
        self._opt_func = opt_func
        self._constraints = constraints
        self._sols = self._init_solutions(nsols)
        self._b = b
        self._a = a
        self._a_step = a_step
        self._maximize = maximize
        self._best_solutions = []

    def get_solutions(self):
        """return solutions"""
        return self._sols

    def optimize(self, max_iters=100, callback=None):
        """Run the Whale Optimization Algorithm for a specified number of iterations."""
        for generation in range(max_iters):
            # Ensure best_solution_pos is updated before starting current iteration's movement
            ranked_sol = self._rank_solutions()
            best_sol_current_gen = ranked_sol[0]

            new_sols = [best_sol_current_gen]  # Keep best solution

            for s in ranked_sol[1:]:
                if np.random.uniform(0.0, 1.0) > 0.5:
                    A = self._compute_A()
                    if np.linalg.norm(A) < 1.0:
                        new_s = self._encircle(s, best_sol_current_gen, A)
                    else:
                        random_sol = self._sols[np.random.randint(self._sols.shape[0])]
                        new_s = self._search(s, random_sol, A)
                else:
                    new_s = self._attack(s, best_sol_current_gen)

                new_sols.append(self._constrain_solution(new_s))

            self._sols = np.stack(new_sols)
            self._a -= self._a_step  # Reduce exploration factor

            # added based on p2
            if len(self._best_solutions) > 5:
                recent = [f[0] for f in self._best_solutions[-5:]]
                if max(recent) - min(recent) < 1e-4:
                    self._a_step *= 1.2  # Slow down reduction of 'a' to encourage more exploration

            # Optional logging callback
            if callback is not None:
                best_fitness, best_solution = self._best_solutions[-1]
                callback(generation, best_fitness, best_solution)

            # Diminishing Returns Detection
            if len(self._best_solutions) > 10:
                recent = [s[0] for s in self._best_solutions[-10:]]  # Last 10 fitness values
                if np.std(recent) < 1e-3:  # Very little improvement
                    print(f"Convergence detected at generation {generation}. Stopping early.")
                    break


    def _init_solutions(self, nsols):
        """initialize solutions uniform randomly in space"""
        sols_list = []
        for _ in range(nsols):
            sol = []
            for lower, upper in self._constraints:
                sol.append(np.random.uniform(lower, upper))
            sols_list.append(sol)
        return np.array(sols_list)

    # Modified: based on p1
    def _constrain_solution(self, sol):
        # Clamp values within defined bounds
        constrain_s = []
        for c, s in zip(self._constraints, sol):
            s = max(min(s, c[1]), c[0])
            constrain_s.append(s)
        constrained = np.array(constrain_s)
        return self._validate_solution(constrained)  # <- validate after constraining

    def _rank_solutions(self):
        """find best solution"""
        # The opt_func now expects the full array of solutions (self._sols)
        # and returns a list/array of fitness values.
        fitness = self._opt_func(self._sols)
        # Ensure fitness is a numpy array for consistent sorting if not already
        if not isinstance(fitness, np.ndarray):
            fitness = np.array(fitness)

        sol_fitness = [(f, s) for f, s in zip(fitness, self._sols)]

        #best solution is at the front of the list
        ranked_sol = list(sorted(sol_fitness, key=lambda x:x[0], reverse=self._maximize))
        self._best_solutions.append(ranked_sol[0])

        return [s[1] for s in ranked_sol] # Return only the solutions (positions)

    def print_best_solutions(self):
        print('\n--- Final Best Solution ---')
        print('([fitness], [solution])')
        # The best solution is the one with the lowest fitness (or highest if maximize is True)
        overall_best_sol = sorted(self._best_solutions, key=lambda x:x[0], reverse=self._maximize)[0]
        print(overall_best_sol)
        return overall_best_sol # Return the overall best solution for further processing

    def _compute_A(self):
        # A should be same dimension as solution (N variables)
        r = np.random.uniform(0.0, 1.0, size=len(self._constraints))
        return (2.0*self._a*r)-self._a # np.multiply is redundant here

    def _compute_C(self):
        # C should be same dimension as solution (N variables)
        return 2.0*np.random.uniform(0.0, 1.0, size=len(self._constraints))

    def _encircle(self, sol, best_sol, A):
        # D is now a vector (element-wise difference)
        D = self._encircle_D_vector(sol, best_sol)
        return best_sol - np.multiply(A, D)

    def _encircle_D_vector(self, sol, best_sol):
        C = self._compute_C()
        # This calculates D as a vector (element-wise difference)
        return np.abs(np.multiply(C, best_sol) - sol)

    def _search(self, sol, rand_sol, A):
        # D is now a vector (element-wise difference)
        D = self._search_D_vector(sol, rand_sol)
        return rand_sol - np.multiply(A, D)

    def _search_D_vector(self, sol, rand_sol):
        C = self._compute_C()
        # This calculates D as a vector (element-wise difference)
        return np.abs(np.multiply(C, rand_sol) - sol)

    def _attack(self, sol, best_sol):
        # D is now a vector (element-wise difference)
        D = np.abs(best_sol - sol)
        # L should be same dimension as solution
        L = np.random.uniform(-1.0, 1.0, size=len(self._constraints))
        # Ensure element-wise multiplication
        return np.multiply(np.multiply(D,np.exp(self._b*L)), np.cos(2.0*np.pi*L))+best_sol

    # from p1
    def _validate_solution(self, sol):
        # Normalize land % to sum to 100
        land_percent = sol[:3]
        total = sum(land_percent)
        if total > 0:
            normed = [p / total * 100 for p in land_percent]
        else:
            normed = [100 / 3] * 3  # fallback to even split
        # Round crop indices
        rounded_crop_indices = [round(x) for x in sol[3:]]
        return np.array(normed + rounded_crop_indices)