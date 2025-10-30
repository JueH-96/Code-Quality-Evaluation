def solve():
    import sys
    input_data = sys.stdin.read().strip().split()
    t = int(input_data[0])
    idx = 1
    
    answers = []
    for _ in range(t):
        n, k = map(int, input_data[idx:idx+2])
        idx += 2
        s = input_data[idx]
        idx += 1
        
        # Count frequencies:
        freq = [0]*26
        for c in s:
            freq[ord(c) - ord('a')] += 1
        
        # Count how many letters have odd occurrences
        odd_count = sum(f % 2 for f in freq)
        
        # The palindrome we want to form has length = n - k
        # final_parity = (n - k) % 2
        # Minimal removals to fix the odd_count is max(0, odd_count - final_parity)
        # We call that delta. Then leftover = k - delta
        # We can only remove leftover in pairs so leftover must be >= 0 and even.
        
        final_parity = (n - k) % 2
        delta = max(0, odd_count - final_parity)
        leftover = k - delta
        
        if leftover >= 0 and leftover % 2 == 0:
            answers.append("YES")
        else:
            answers.append("NO")
    
    print("
".join(answers))

def main():
    solve()

if __name__ == "__main__":
    main()