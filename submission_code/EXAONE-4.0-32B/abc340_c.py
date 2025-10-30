from collections import defaultdict

def main():
    N = int(input().strip())
    freq = defaultdict(int)
    freq[N] = 1
    total_cost = 0
    while True:
        new_freq = defaultdict(int)
        found_ge2 = False
        for x, cnt in freq.items():
            if x < 2:
                new_freq[x] += cnt
            else:
                found_ge2 = True
                total_cost += x * cnt
                a = x // 2
                b = (x + 1) // 2
                new_freq[a] += cnt
                new_freq[b] += cnt
        
        if not found_ge2:
            break
        freq = new_freq
    
    print(total_cost)

if __name__ == "__main__":
    main()