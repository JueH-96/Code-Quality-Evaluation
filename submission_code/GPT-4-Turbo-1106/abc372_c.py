def count_abc_substrings(s):
    return sum(1 for i in range(len(s) - 2) if s[i:i+3] == "ABC")

def process_queries(n, q, s, queries):
    for x, c in queries:
        s = s[:x-1] + c + s[x:]
        print(count_abc_substrings(s))

def main():
    n, q = map(int, input().split())
    s = input().strip()
    queries = [tuple(input().split()) for _ in range(q)]
    queries = [(int(x), c) for x, c in queries]
    process_queries(n, q, s, queries)

if __name__ == "__main__":
    main()