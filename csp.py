from collections import Counter

# Patients and their preferences
preferences = {
    'A': ['ICU', 'General'],
    'B': ['General', 'Private'],
    'C': ['Emergency', 'General'],
    'D': ['ICU', 'Private'],
    'E': ['General', 'Emergency'],
    'F': ['Emergency', 'Private']
}

# Bed capacities
bed_capacities = {
    'ICU': 2,
    'General': 3,
    'Private': 1,
    'Emergency': 2
}

# Patient conflicts
conflicts = [('A', 'B'), ('C', 'D')]

# Track recursive calls
recursive_calls = 0

# Backtracking solver
def is_valid(assignment, patient, bed):
    # Check bed capacity
    count = Counter(assignment.values())
    if count[bed] >= bed_capacities[bed]:
        return False
    
    # Check conflicts
    for p1, p2 in conflicts:
        if (patient == p1 and p2 in assignment and assignment[p2] == bed) or \
           (patient == p2 and p1 in assignment and assignment[p1] == bed):
            return False
    
    return True

def backtrack(assignment, patients):
    global recursive_calls
    recursive_calls += 1

    if len(assignment) == len(patients):
        return assignment
    
    patient = patients[len(assignment)]
    for bed in preferences[patient]:
        if is_valid(assignment, patient, bed):
            assignment[patient] = bed
            result = backtrack(assignment, patients)
            if result:
                return result
            del assignment[patient]  # Backtrack
    
    return None

# Run CSP solver
patients = list(preferences.keys())
solution = backtrack({}, patients)

# Output
print("Final bed allocation:", solution)
print("Total recursive calls:", recursive_calls)