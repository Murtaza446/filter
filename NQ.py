import sys
import time
from ortools.sat.python import cp_model


class NQueenSolutionPrinter(cp_model.CpSolverSolutionCallback):
    """Print solutions for the N-Queens problem."""
    def __init__(self, queens):
        cp_model.CpSolverSolutionCallback.__init__(self)
        self.__queens = queens
        self.__solution_count = 0

    def on_solution_callback(self):
        self.__solution_count += 1
        solution = [self.Value(self.__queens[i]) for i in range(len(self.__queens))]
        print(f"Solution {self.__solution_count}: {solution}")

    def solution_count(self):
        return self.__solution_count


def solve_n_queens(board_size):
    # Create the model
    model = cp_model.CpModel()

    # Define variables
    queens = [model.NewIntVar(0, board_size - 1, f"x_{i}") for i in range(board_size)]

    # Add constraints
    # All rows must be different
    model.AddAllDifferent(queens)

    # No two queens can be on the same diagonal
    model.AddAllDifferent([queens[i] + i for i in range(board_size)])
    model.AddAllDifferent([queens[i] - i for i in range(board_size)])

    # Solve the model
    solver = cp_model.CpSolver()
    solution_printer = NQueenSolutionPrinter(queens)
    solver.parameters.enumerate_all_solutions = True
    solver.Solve(model, solution_printer)


if __name__ == "__main__":
    board_size = 8  # You can change the board size here
    solve_n_queens(board_size)