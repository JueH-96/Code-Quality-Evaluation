def main():
    import sys
    data = sys.stdin.read().split()
    if not data:
        return
    n = int(data[0])
    people = []
    pos = 1
    # Read each person's data and calculate their success rate.
    for i in range(1, n+1):
        A = int(data[pos])
        B = int(data[pos+1])
        pos += 2
        total = A + B
        # Calculate success rate as A/(A+B)
        rate = A / total
        people.append((rate, i))
        
    # Sort people in descending order of success rate;
    # If two people have the same success rate, sort by their number in ascending order.
    people.sort(key=lambda person: (-person[0], person[1]))
    
    # Output the sorted order of indices.
    sys.stdout.write(" ".join(str(person) for _, person in people))

if __name__ == '__main__':
    main()