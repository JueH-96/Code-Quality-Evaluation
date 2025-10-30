def main():
    N = int(input())
    H = list(map(int, input().split()))

    T = 0
    for health in H:
        # Calculate the number of attacks needed for each enemy
        normal_attacks = (health % 3) + (health // 3)
        T += normal_attacks

    print(T)

if __name__ == "__main__":
    main()