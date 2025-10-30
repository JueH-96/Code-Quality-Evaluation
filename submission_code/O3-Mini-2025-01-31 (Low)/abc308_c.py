def main():
    import sys
    data = sys.stdin.read().strip().split()
    if not data:
        return
    it = iter(data)
    n = int(next(it))
    people = []
    # Read input and store tuple (person_number, A, B)
    for i in range(1, n + 1):
        a = int(next(it))
        b = int(next(it))
        people.append((i, a, b))
    
    # Sort by descending success rate, tie-breaker: ascending person number.
    # Success rate = A / (A+B)
    # We use the key function: (-A/(A+B), i) to sort in descending order by rate
    people.sort(key=lambda x: (-x[1] / (x[1] + x[2]), x[0]))
    
    # Produce result as space-separated string of person numbers.
    result = " ".join(str(person[0]) for person in people)
    sys.stdout.write(result)

if __name__ == '__main__':
    main()