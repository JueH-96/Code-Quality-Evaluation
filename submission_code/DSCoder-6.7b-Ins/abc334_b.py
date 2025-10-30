def main():
    A, M, L, R = map(int, input().split())

    # Calculate the minimum and maximum possible coordinates of the trees
    min_tree = ((L + M - 1) // M) * M
    max_tree = (R // M) * M

    # Calculate the number of trees
    num_trees = (max_tree - min_tree) // M + 1

    print(num_trees)

if __name__ == "__main__":
    main()