def main():
    import sys
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr +=1
    M = int(input[ptr])
    ptr +=1
    A = list(map(int, input[ptr:ptr+M]))
    
    counts = [0]*(N+1)
    max_count = 0
    current_winner = 0
    
    for a in A:
        counts[a] +=1
        if counts[a] > max_count:
            max_count = counts[a]
            current_winner = a
        elif counts[a] == max_count:
            if a < current_winner:
                current_winner = a
        print(current_winner)
        
if __name__ == '__main__':
    main()