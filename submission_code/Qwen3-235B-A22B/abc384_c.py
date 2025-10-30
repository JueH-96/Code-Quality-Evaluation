def main():
    import sys
    a, b, c, d, e = map(int, sys.stdin.readline().split())
    scores = [a, b, c, d, e]
    letters = ['A', 'B', 'C', 'D', 'E']
    participants = []
    
    for mask in range(1, 1 << 5):
        name = ''.join(letters[i] for i in range(5) if (mask & (1 << i)))
        score = sum(scores[i] for i in range(5) if (mask & (1 << i)))
        participants.append((-score, name))
    
    participants.sort()
    
    for p in participants:
        print(p[1])

if __name__ == "__main__":
    main()