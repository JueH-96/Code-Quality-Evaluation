def main():
    import sys
    input = sys.stdin.readline
    
    N = int(input().strip())
    names = []
    total_rating = 0
    
    for _ in range(N):
        s, c = input().split()
        c = int(c)
        names.append(s)
        total_rating += c
    
    names.sort()
    winner_index = total_rating % N
    print(names[winner_index])

if __name__ == "__main__":
    main()