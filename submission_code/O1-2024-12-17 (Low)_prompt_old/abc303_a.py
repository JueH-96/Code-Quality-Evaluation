def solve():
    import sys
    
    data = sys.stdin.read().strip().split()
    N = int(data[0])
    S = data[1]
    T = data[2]
    
    # Define which pairs of characters are considered similar
    similar_pairs = {('0', 'o'), ('o', '0'), ('1', 'l'), ('l', '1')}
    
    for i in range(N):
        s_char = S[i]
        t_char = T[i]
        # Check if they are the same or if they match one of the similar pairs
        if s_char != t_char and (s_char, t_char) not in similar_pairs:
            print("No")
            return
    
    print("Yes")

def main():
    solve()

if __name__ == "__main__":
    main()