import sys
input = sys.stdin.read

def solve():
    data = input().split()
    N = int(data[0])
    A = list(map(int, data[1:]))

    # Dictionary to store the occurrences of each number
    occurrences = {}

    # Iterate through the list and store the indices of occurrences
    for index, value in enumerate(A):
        if value in occurrences:
            occurrences[value].append(index + 1)
        else:
            occurrences[value] = [index + 1]

    # Calculate f(i) for each i
    f = {}
    for key, indices in occurrences.items():
        # The middle occurrence is the second index in the sorted list of indices
        f[key] = indices[1]

    # Sort the numbers based on f(i)
    sorted_numbers = sorted(f.keys(), key=lambda x: f[x])

    # Print the result
    print(" ".join(map(str, sorted_numbers)))

if __name__ == "__main__":
    solve()