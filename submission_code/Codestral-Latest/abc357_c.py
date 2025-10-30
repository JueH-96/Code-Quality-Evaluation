def generate_carpet(N):
    if N == 0:
        return ["#"]

    def create_level_k_carpet(K):
        if K == 0:
            return ["#"]

        smaller_carpet = create_level_k_carpet(K - 1)
        size = 3 ** (K - 1)
        white_block = ["." * size for _ in range(size)]

        top_bottom = [smaller_carpet * 3]
        middle = [smaller_carpet + white_block + smaller_carpet]

        carpet = top_bottom + middle + top_bottom
        return ["".join(row) for row in zip(*carpet)]

    return create_level_k_carpet(N)

def main():
    import sys
    input = sys.stdin.read
    data = input().strip()
    N = int(data)
    carpet = generate_carpet(N)
    for row in carpet:
        print(row)

if __name__ == "__main__":
    main()