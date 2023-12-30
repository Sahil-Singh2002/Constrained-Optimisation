import numpy as np

#Data points
data_points = np.array([
    [1.5, 1.0, 1.8],
    [1.9, 1.9, 1.3],
    [1.8, 1.7, 1.3],
    [1.2, 1.8, 1.2],
    [1.0, 1.4, 1.6],
    [1.4, 1.1, 1.9],
    [1.8, 1.0, 1.1],
    [1.3, 1.3, 1.8],
    [1.7, 1.9, 1.5],
    [1.3, 1.2, 1.3]
])

#Objective function
def objective_function(x, y, data_points):
  a_i = data_points
  sum = np.sum((np.dot(a_i,x) - y)**2 / np.dot(x, x))
  return sum

#Optimal hyperplane
def optimal_hyperplane(data_points):
    a_i = data_points
    A = np.cov(a_i, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(A)
    smallest_eigenvalue_index = np.argmin(eigenvalues)
    x = eigenvectors[:, smallest_eigenvalue_index]
    y = np.mean(a_i.dot(x))
    return x, y

#Gradient Method on the Rayleigh quotient
def gradient_method(data_points, num_iterations=10**4, t=0.000001):
    a_i = data_points
    x, y = optimal_hyperplane(a_i)
    for _ in range(num_iterations):

        residuals = np.dot(a_i, x) - y
        gradient = 2 * (np.dot(a_i.T, residuals) / np.dot(x, x))
        x = x - t * gradient
        y = np.mean(a_i.dot(x))
    return x, y

def f_range(data_points):
    a_i = data_points
    x, y = gradient_method(a_i)
    f_value = objective_function(x, y, a_i)
    while (f_value <= 0.45) or (f_value >= 0.46):
        x, y = gradient_method(a_i)
        f_value = objective_function(x,y, a_i)

    return x, y, f_value

x, y, f_value = f_range(data_points)
print("#------------------------------------------------------#")
print("Optimal 'x' vector:", x)
print("#------------------------------------------------------#")
print("Optimal 'y' value: ", y)
print("#------------------------------------------------------#")
print(f"Objective function value 0.45<f(x,y)={round(f_value,4)}<0.46:")
print("#------------------------------------------------------#")

import numpy as np

def weiszfeld_algorithm(anchors, weights):
    x_k = np.array([0,0])

    for _ in range(10**4):

        # Distances and Weights for each anchor
        w_d = weights / np.linalg.norm(x_k - anchors, axis=1)

        # Avoid division by zero
        w_d[np.isinf(w_d)] = 0

        #Weiszfeld iteration formula
        x_k = np.sum(anchors * w_d[:, np.newaxis], axis=0) / np.sum(w_d)

        # Check convergence based on gradient
        gradient = np.sum(weights[:, np.newaxis] * (x_k - anchors) / np.linalg.norm(x_k - anchors, axis=1)[:, np.newaxis], axis=0)

        if np.linalg.norm(gradient) < 1e-12:
          break

        x_kone = x_k

    # Hessian
    hessian = np.zeros((2, 2))
    for i in range(len(anchors)):
        hessian = hessian + weights[i]*(np.outer(x_kone - anchors[i], x_kone - anchors[i]) / np.linalg.norm(x_kone - anchors, axis=1)[i]**3)

    # Positive definite
    is_positive_definite = np.all(np.linalg.eigvals(hessian) > 0)

    x = x_kone
    return x, is_positive_definite, _ + 1

# Given data
anchors = np.array([[1, 2], [3, 0], [3, 1], [2, 3]])
weights = 2*np.array([4, 1, 2, 3])
cost_per_kilometer = 1.0  # Cost in pounds

# Run the Weiszfeld algorithm
apartment_coordinates, is_positive_definite, iterations = weiszfeld_algorithm(anchors, weights)

#Convert to SI units and make it into integers

SI_apartment_coordinates = (apartment_coordinates*1000)
SI_apartment_coordinates = [int(round(SI_apartment_coordinates[0])),int(round(SI_apartment_coordinates[1]))]


weekly_transportation_expenses = np.sum(weights * np.linalg.norm(apartment_coordinates - anchors, axis=1) * cost_per_kilometer)

print("Apartment Coordinates (meters):",SI_apartment_coordinates)
print("Weekly Transportation Expenses: £", round(weekly_transportation_expenses,2))
print("Number of iterations:", iterations)
print("Is Hessian Positive Definite?", is_positive_definite)

# Check if below budget

if weekly_transportation_expenses< 22.79:
  print("You are under budget")
else:
  print("weekly transport expenses above Constraint violation: £",22.79)
