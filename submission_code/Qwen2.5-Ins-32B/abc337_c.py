import sys
input = sys.stdin.read

def solve(n, a):
    # Create a dictionary to store the list of people behind each person
    behind = {i: [] for i in range(1, n + 1)}
    # Create a list to store the front person
    front = []
    # Populate the behind dictionary and find the front person
    for i, ai in enumerate(a, 1):
        if ai == -1:
            front.append(i)
        else:
            behind[ai].append(i)
    # Initialize the result list with the front person
    result = front
    # Traverse the behind dictionary to find the order of people
    while result[-1] in behind:
        result.extend(behind[result[-1]])
    return result

def main():
    data = input().split()
    n = int(data[0])
    a = list(map(int, data[1:]))
    result = solve(n, a)
    print(' '.join(map(str, result)))

if __name__ == "__main__":
    main()