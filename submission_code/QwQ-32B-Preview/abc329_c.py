def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    S = data[1]
    
    max_streak = {}
    if N >= 1:
        current_char = S[0]
        current_streak = 1
        max_streak[current_char] = 1
        for i in range(1, N):
            if S[i] == current_char:
                current_streak += 1
                if current_streak > max_streak.get(current_char, 0):
                    max_streak[current_char] = current_streak
            else:
                if current_streak > max_streak.get(current_char, 0):
                    max_streak[current_char] = current_streak
                current_char = S[i]
                current_streak = 1
                if current_streak > max_streak.get(current_char, 0):
                    max_streak[current_char] = current_streak
        # After loop ends, update the last character's streak
        if current_streak > max_streak.get(current_char, 0):
            max_streak[current_char] = current_streak
    
    total_unique_substrings = sum(max_streak.values())
    print(total_unique_substrings)

if __name__ == "__main__":
    main()