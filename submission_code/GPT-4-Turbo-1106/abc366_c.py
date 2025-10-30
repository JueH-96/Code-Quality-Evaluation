from collections import Counter

def process_queries(Q, queries):
    bag = Counter()
    for query in queries:
        if query[0] == 1:
            bag[query[1]] += 1
        elif query[0] == 2:
            bag[query[1]] -= 1
            if bag[query[1]] == 0:
                del bag[query[1]]
        elif query[0] == 3:
            print(len(bag))

def main():
    Q = int(input().strip())
    queries = []
    for _ in range(Q):
        query = list(map(int, input().strip().split()))
        queries.append(query)
    process_queries(Q, queries)

if __name__ == "__main__":
    main()