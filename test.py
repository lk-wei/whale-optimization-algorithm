import numpy as np
import random
from collections import defaultdict


# --- 1. Problem Definition ---
class UrbanAgProblem:
    def __init__(self, total_land_sqm=1000):  # Default to 1000 sqm if not specified
        self.total_land_sqm = total_land_sqm

        # Define bounds for each variable
        # [Hydroponics %, Container %, In-Ground %, Hydroponics Crop, Container Crop]
        # Percentages are 0-100. Crops are 0 or 1 for this simple example.
        self.bounds = [
            (0, 100),  # x1: Hydroponics %
            (0, 100),  # x2: Container %
            (0, 100),  # x3: In-Ground %
            (0, 1),  # c1: Hydroponics Crop (0=Leafy Greens, 1=Herbs)
            (0, 1)  # c2: Container Crop (0=Root Vegetables, 1=Tomatoes)
        ]
        self.num_variables = len(self.bounds)

        # --- IMPORTANT: Realistic Data for Crops and Systems (replace with your actual data) ---
        # Yield (kg/sqm/year), Water Consumption (liters/sqm/year), Cost (USD/sqm/year)
        # Using slightly more aggressive (but still simple) data to encourage yield
        self.hydro_data = {
            0: {'name': 'Leafy Greens (Hydro)', 'yield_per_sqm': 40, 'water_per_sqm': 40, 'cost_per_sqm': 15},
            1: {'name': 'Herbs (Hydro)', 'yield_per_sqm': 20, 'water_per_sqm': 35, 'cost_per_sqm': 14}
        }
        self.container_data = {
            0: {'name': 'Root Vegetables (Container)', 'yield_per_sqm': 15, 'water_per_sqm': 120, 'cost_per_sqm': 7},
            1: {'name': 'Tomatoes (Container)', 'yield_per_sqm': 18, 'water_per_sqm': 150, 'cost_per_sqm': 8}
        }
        self.inground_data = {
            'name': 'General In-Ground', 'yield_per_sqm': 10, 'water_per_sqm': 200, 'cost_per_sqm': 4
        }

    def evaluate(self, solution):
        """
        Evaluates a given solution (combination of variables) and calculates
        total yield, total water consumption, and total cost.
        """
        x1, x2, x3, c1_idx, c2_idx = solution

        # --- 1. Handle Percentage Constraint and Convert to Area ---
        # Normalize percentages if they sum to more than 100%
        total_percent = x1 + x2 + x3
        if total_percent > 100:
            factor = 100 / total_percent
            x1 *= factor
            x2 *= factor
            x3 *= factor

        # Ensure percentages are within bounds
        x1 = np.clip(x1, 0, 100)
        x2 = np.clip(x2, 0, 100)
        x3 = np.clip(x3, 0, 100)

        area_hydro = (x1 / 100) * self.total_land_sqm
        area_container = (x2 / 100) * self.total_land_sqm
        area_inground = (x3 / 100) * self.total_land_sqm

        # --- 2. Convert Categorical Crop Choices to Integers ---
        # Use round and then int for robustness, ensuring it picks valid index
        c1_idx = int(round(c1_idx))
        c2_idx = int(round(c2_idx))

        # --- 3. Calculate Metrics Based on Allocations and Crops ---
        total_yield = 0
        total_water = 0
        total_cost = 0

        # Hydroponics calculations
        hydro_crop_data = self.hydro_data.get(c1_idx, self.hydro_data[0])  # Default to Leafy Greens if index invalid
        total_yield += area_hydro * hydro_crop_data['yield_per_sqm']
        total_water += area_hydro * hydro_crop_data['water_per_sqm']
        total_cost += area_hydro * hydro_crop_data['cost_per_sqm']

        # Container/Raised Bed calculations
        container_crop_data = self.container_data.get(c2_idx,
                                                      self.container_data[0])  # Default to Root Veg if index invalid
        total_yield += area_container * container_crop_data['yield_per_sqm']
        total_water += area_container * container_crop_data['water_per_sqm']
        total_cost += area_container * container_crop_data['cost_per_sqm']

        # In-Ground calculations
        total_yield += area_inground * self.inground_data['yield_per_sqm']
        total_water += area_inground * self.inground_data['water_per_sqm']
        total_cost += area_inground * self.inground_data['cost_per_sqm']

        # --- 4. Define Fitness (Objective Function) ---
        # To MAXIMIZE Total Yield:
        # We define fitness as -total_yield, because the WOA algorithm MINIMIZES fitness.
        # Minimizing -total_yield is equivalent to maximizing total_yield.

        fitness = -total_yield

        # Optionally, you could add very small penalties for water/cost if you want
        # to break ties when multiple solutions yield the same maximum,
        # but for pure maximization of yield, just focus on yield.
        # Example with slight penalty:
        # fitness = -total_yield + (total_water * 0.0001) + (total_cost * 0.00001)
        # Make sure the penalties are much, much smaller than the yield impact.

        return fitness, total_yield, total_water, total_cost


# --- 2. Whale Optimization Algorithm ---
class WOA:
    def __init__(self, problem, n_whales=30, max_iter=100):
        self.problem = problem
        self.n_whales = n_whales
        self.max_iter = max_iter
        self.whales = self._initialize_whales()
        self.best_solution_pos = None
        self.best_solution_fitness = float('inf')  # Minimization problem

        # To store results for analysis
        self.history = defaultdict(list)

    def _initialize_whales(self):
        whales = []
        for _ in range(self.n_whales):
            pos = []
            for lower_bound, upper_bound in self.problem.bounds:
                pos.append(lower_bound + random.random() * (upper_bound - lower_bound))
            whales.append(np.array(pos))
        return whales

    def _apply_bounds(self, position):
        for i, (lower, upper) in enumerate(self.problem.bounds):
            position[i] = np.clip(position[i], lower, upper)
        return position

    def optimize(self):
        for t in range(self.max_iter):
            a = 2 - t * (2 / self.max_iter)  # 'a' decreases linearly from 2 to 0

            for i in range(self.n_whales):
                # Update best solution found so far
                current_fitness, _, _, _ = self.problem.evaluate(self.whales[i])
                if current_fitness < self.best_solution_fitness:
                    self.best_solution_fitness = current_fitness
                    self.best_solution_pos = np.copy(self.whales[i])

            # Ensure best_solution_pos is not None before using it
            if self.best_solution_pos is None:
                # If no solution has been found yet (e.g., first iteration, all initial solutions invalid)
                # This should ideally not happen if initial solutions are generated within bounds.
                # As a fallback, use a random whale for the first step if best_solution_pos is truly None
                self.best_solution_pos = self.whales[np.random.randint(self.n_whales)]
                self.best_solution_fitness = self.problem.evaluate(self.best_solution_pos)[0]

            for i in range(self.n_whales):
                r1, r2 = random.random(), random.random()
                A = 2 * a * r1 - a
                C = 2 * r2
                p = random.random()  # For choosing between shrinking encircling and spiral updating

                # Encircling Prey or Searching for Prey
                if abs(A) < 1:  # Shrinking Encircling Mechanism (Exploitation)
                    D = abs(C * self.best_solution_pos - self.whales[i])
                    self.whales[i] = self.best_solution_pos - A * D
                else:  # Search for Prey (Exploration)
                    rand_whale_idx = random.randint(0, self.n_whales - 1)
                    rand_whale_pos = self.whales[rand_whale_idx]
                    D = abs(C * rand_whale_pos - self.whales[i])
                    self.whales[i] = rand_whale_pos - A * D

                # Spiral Updating Position (Exploitation)
                # Only apply spiral if p < 0.5 for the exploitation phase
                if p < 0.5:
                    b = 1  # Constant for defining the shape of the logarithmic spiral
                    l = (a - 1) * random.random() + 1  # Random number in [-1, 1]
                    D_prime = abs(self.best_solution_pos - self.whales[i])
                    self.whales[i] = D_prime * np.exp(b * l) * np.cos(2 * np.pi * l) + self.best_solution_pos

                # Apply bounds after position update
                self.whales[i] = self._apply_bounds(self.whales[i])

            # Store iteration results
            self.history['best_fitness'].append(self.best_solution_fitness)
            _, current_yield, current_water, current_cost = self.problem.evaluate(self.best_solution_pos)
            self.history['best_yield'].append(current_yield)
            self.history['best_water'].append(current_water)
            self.history['best_cost'].append(current_cost)

        return self.best_solution_pos, self.best_solution_fitness


# --- 3. Run the Optimization ---
if __name__ == "__main__":
    # Define your total land area (e.g., 500 square meters)
    total_urban_land = 500

    problem = UrbanAgProblem(total_land_sqm=total_urban_land)
    woa = WOA(problem, n_whales=100, max_iter=500)  # Increased whales and iterations for better exploration

    print(f"Starting WOA for urban agriculture planning with {total_urban_land} sqm land (MAXIMIZING YIELD)...")
    best_solution, best_fitness = woa.optimize()

    # --- Interpret Results ---
    print("\n--- Optimization Complete ---")
    # Best fitness for maximization of yield will be the largest negative number
    print(f"Best found fitness (lower is better, directly corresponds to -Yield): {best_fitness:.4f}")

    # Re-evaluate the best solution to get the individual metrics
    fitness, final_yield, final_water, final_cost = problem.evaluate(best_solution)

    x1_perc, x2_perc, x3_perc, c1_idx_raw, c2_idx_raw = best_solution

    # Round percentages and ensure they sum close to 100
    total_percent_raw = x1_perc + x2_perc + x3_perc
    if total_percent_raw > 0:  # Avoid division by zero
        factor = 100 / total_percent_raw
        x1_perc_adj = x1_perc * factor
        x2_perc_adj = x2_perc * factor
        x3_perc_adj = x3_perc * factor
    else:  # If all are zero, set to zero
        x1_perc_adj, x2_perc_adj, x3_perc_adj = 0, 0, 0

    x1_perc_adj = np.round(x1_perc_adj, 2)
    x2_perc_adj = np.round(x2_perc_adj, 2)
    x3_perc_adj = np.round(x3_perc_adj, 2)

    # Convert area percentages to actual square meters
    area_hydro = (x1_perc_adj / 100) * total_urban_land
    area_container = (x2_perc_adj / 100) * total_urban_land
    area_inground = (x3_perc_adj / 100) * total_urban_land

    # Convert crop indices back to names
    c1_name = problem.hydro_data[int(round(c1_idx_raw))]['name'] if int(
        round(c1_idx_raw)) in problem.hydro_data else "Unknown Hydro Crop"
    c2_name = problem.container_data[int(round(c2_idx_raw))]['name'] if int(
        round(c2_idx_raw)) in problem.container_data else "Unknown Container Crop"

    print("\n--- Optimal Urban Agriculture Plan (Max Yield Focus) ---")
    print(f"Total Land Available: {total_urban_land} sqm")
    print(
        f"Total Area Allocated: {area_hydro + area_container + area_inground:.2f} sqm (Sum of percentages adjusted to 100%)")
    print(f"------------------------------------")
    print(f"Hydroponics:")
    print(f"  - Percentage of Land: {x1_perc_adj:.2f}% ({area_hydro:.2f} sqm)")
    print(f"  - Primary Crop: {c1_name}")
    print(f"Container/Raised Beds:")
    print(f"  - Percentage of Land: {x2_perc_adj:.2f}% ({area_container:.2f} sqm)")
    print(f"  - Primary Crop: {c2_name}")
    print(f"In-Ground Soil-Based Farming:")
    print(f"  - Percentage of Land: {x3_perc_adj:.2f}% ({area_inground:.2f} sqm)")

    print("\n--- Predicted Performance Metrics ---")
    print(f"Total Annual Yield: {final_yield:.2f} kg")
    print(f"Total Annual Water Consumption: {final_water:.2f} liters")
    print(f"Total Annual Cost: ${final_cost:.2f}")

    # You can also plot the history to see convergence
    # import matplotlib.pyplot as plt
    # plt.figure(figsize=(10, 6))
    # plt.plot(woa.history['best_fitness'])
    # plt.title('Best Fitness Over Iterations (Minimizing -Yield)')
    # plt.xlabel('Iteration')
    # plt.ylabel('Fitness')
    # plt.grid(True)
    # plt.show()