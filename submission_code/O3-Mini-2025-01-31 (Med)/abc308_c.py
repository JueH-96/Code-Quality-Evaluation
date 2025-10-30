def main():
    import sys

    data = sys.stdin.read().split()
    if not data:
        return

    n = int(data[0])
    persons = []
    pos = 1
    for i in range(1, n+1):
        A = int(data[pos])
        B = int(data[pos+1])
        pos += 2
        # Compute success rate as A / (A+B)
        ratio = A / (A + B)
        persons.append((i, ratio))
        
    # Sort descending by success ratio.
    # Ties are broken by ascending order of the person's number.
    persons.sort(key=lambda person: (-person[1], person[0]))

    # Output the ordered IDs space-separated on one line.
    sys.stdout.write(" ".join(str(person[0]) for person in persons))

if __name__ == '__main__':
    main()