def main():
    import sys
    input = sys.stdin.read
    data = input().splitlines()
    
    N, M = map(int, data[0].split())
    families = {}
    results = []
    
    for i in range(1, M + 1):
        A_i, B_i = data[i].split()
        A_i = int(A_i)
        
        if B_i == 'M':
            if A_i not in families:
                families[A_i] = True  # Mark the family as having a Taro
                results.append("Yes")
            else:
                results.append("No")
        else:
            results.append("No")
    
    print("
".join(results))

if __name__ == "__main__":
    main()