import sys

class Person:
    """
    Represents a person with their coin toss statistics.
    Implements a custom comparison logic for sorting based on success rate.
    """
    def __init__(self, A, B, index):
        self.A = A
        self.total = A + B
        self.index = index

    def __lt__(self, other):
        """
        Custom comparison for sorting. A person 'self' is "less than" 'other'
        if it should come before 'other' in the final sorted list.
        
        Sorting criteria:
        1. Descending order of success rate (A / total).
        2. Ascending order of index (for ties in success rate).

        To avoid floating-point inaccuracies, we compare rates using
        integer cross-multiplication. The comparison:
        self.A / self.total > other.A / other.total
        is equivalent to:
        self.A * other.total > other.A * self.total
        """
        # Calculate the cross-multiplication products.
        val_self = self.A * other.total
        val_other = other.A * self.total

        # Primary sort key: success rate (descending)
        if val_self != val_other:
            return val_self > val_other
        
        # Secondary sort key: index (ascending)
        else:
            return self.index < other.index

def solve():
    """
    Reads input, performs the sorting, and prints the result.
    """
    # Use fast I/O for large inputs
    input = sys.stdin.readline

    # Read the number of people
    try:
        N = int(input())
    except (ValueError, IndexError):
        return # Handle empty input gracefully

    # Read data for each person and create a list of Person objects
    people = []
    for i in range(1, N + 1):
        line = input()
        if not line:
            break
        A, B = map(int, line.split())
        people.append(Person(A, B, i))

    # Sort the list in-place using the custom __lt__ method
    people.sort()

    # Extract the sorted indices
    sorted_indices = [p.index for p in people]

    # Print the result as space-separated values
    print(*sorted_indices)

solve()