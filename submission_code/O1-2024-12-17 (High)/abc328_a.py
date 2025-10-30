def main():
    import sys
    data = sys.stdin.read().strip().split()
    N, X = map(int, data[:2])
    scores = list(map(int, data[2:2+N]))
    
    total_score = sum(s for s in scores if s <= X)
    print(total_score)

# Do not forget to call main()
main()