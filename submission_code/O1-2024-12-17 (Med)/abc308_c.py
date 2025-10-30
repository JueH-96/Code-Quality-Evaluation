def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    N = int(input_data[0])
    people = []
    idx = 1
    for i in range(1, N+1):
        A = int(input_data[idx]); B = int(input_data[idx+1])
        idx += 2
        # Compute success rate as a float
        ratio = A / (A + B)
        # Store (person_number, ratio), will sort descending by ratio, tie by ascending person_number
        people.append((i, ratio))
    
    # Sort by negative ratio (for descending), then by person_number ascending
    people.sort(key=lambda x: (-x[1], x[0]))
    
    # Output the sorted order of person numbers
    print(*[p[0] for p in people])

# Don't forget to call main!
main()