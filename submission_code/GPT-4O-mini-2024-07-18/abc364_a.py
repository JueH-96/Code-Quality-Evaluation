def can_eat_all_dishes(n, dishes):
    for i in range(1, n):
        if dishes[i] == "sweet" and dishes[i - 1] == "sweet":
            return "No"
    return "Yes"

if __name__ == "__main__":
    import sys
    input = sys.stdin.read
    data = input().splitlines()
    
    N = int(data[0])
    S = data[1:N + 1]
    
    result = can_eat_all_dishes(N, S)
    print(result)