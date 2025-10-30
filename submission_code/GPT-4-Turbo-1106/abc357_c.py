def generate_carpet(n):
    if n == 0:
        return ['#']
    else:
        smaller_carpet = generate_carpet(n - 1)
        top_and_bottom = [line * 3 for line in smaller_carpet]
        middle = [line + '.' * len(line) + line for line in smaller_carpet]
        return top_and_bottom + middle + top_and_bottom

def print_carpet(n):
    carpet = generate_carpet(n)
    for line in carpet:
        print(line)

if __name__ == "__main__":
    N = int(input().strip())
    print_carpet(N)