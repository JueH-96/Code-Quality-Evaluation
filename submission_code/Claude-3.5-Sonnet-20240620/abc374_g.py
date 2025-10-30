# YOUR CODE HERE
def min_ng_list(used_names):
    def can_merge(s1, s2):
        return any(s1.endswith(s2[:i]) or s2.startswith(s1[-i:]) for i in range(1, min(len(s1), len(s2)) + 1))

    def merge(s1, s2):
        for i in range(min(len(s1), len(s2)), 0, -1):
            if s1.endswith(s2[:i]):
                return s1 + s2[i:]
            if s2.startswith(s1[-i:]):
                return s2 + s1[i:]
        return s1 + s2

    ng_list = list(used_names)

    while True:
        merged = False
        for i in range(len(ng_list)):
            for j in range(i + 1, len(ng_list)):
                if can_merge(ng_list[i], ng_list[j]):
                    merged_string = merge(ng_list[i], ng_list[j])
                    ng_list = [s for k, s in enumerate(ng_list) if k not in (i, j)] + [merged_string]
                    merged = True
                    break
            if merged:
                break
        if not merged:
            break

    return len(ng_list)

N = int(input())
used_names = [input().strip() for _ in range(N)]

result = min_ng_list(used_names)
print(result)