import sys

def main():
    N, X, Y = map(int, sys.stdin.readline().split())
    dishes = []
    for _ in range(N):
        a, b = map(int, sys.stdin.readline().split())
        dishes.append((a, b))
    
    # Sort dishes by A + B, then by A, then B to have a consistent order
    dishes.sort(key=lambda x: (x[0] + x[1], x[0], x[1]))
    
    dp = {(0, 0): 0}
    max_count = 0
    
    for a, b in dishes:
        temp = {}
        # Iterate over a copy of the current dp items
        for (current_a, current_b), count in dp.items():
            # Option to take the current dish
            new_a = current_a + a
            new_b = current_b + b
            if new_a > X or new_b > Y:
                if count + 1 > max_count:
                    max_count = count + 1
            else:
                new_count = count + 1
                key = (new_a, new_b)
                if key in temp:
                    if new_count > temp[key]:
                        temp[key] = new_count
                else:
                    temp[key] = new_count
        # Merge temp into dp
        for key in temp:
            if key in dp:
                if temp[key] > dp[key]:
                    dp[key] = temp[key]
            else:
                dp[key] = temp[key]
    
    # The answer is the maximum between max_count and the max in dp.values()
    max_dp = max(dp.values()) if dp else 0
    answer = max(max_count, max_dp)
    print(answer)

if __name__ == "__main__":
    main()