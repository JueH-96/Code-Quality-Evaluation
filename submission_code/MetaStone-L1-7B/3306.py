class FenwickTree:
    def __init__(self, size):
        self.n = size
        self.tree = [0] * (self.n + 2)  # 1-based indexing

    def update(self, index, delta):
        while index <= self.n:
            self.tree[index] += delta
            index += index & -index

    def query(self, index):
        res = 0
        while index > 0:
            res += self.tree[index]
            index -= index & -index
        return res

def unmarkedSumArray(nums: List[int], queries: List[List[int]]) -> List[int]:
    n = len(nums)
    sorted_list = sorted((v, i) for i, v in enumerate(nums))
    pos_map = {(v, i): idx for idx, (v, i) in enumerate(sorted_list)}

    count_tree = FenwickTree(n)
    sum_tree = FenwickTree(n)

    for idx in range(n):
        v = nums[idx]
        pos = pos_map[(v, idx)]
        count_tree.update(pos + 1, 1)
        sum_tree.update(pos + 1, v)

    marked_sum = 0
    current_sum = sum(nums)
    answers = []

    for query in queries:
        index_i, k_i = query
        sum_k = 0
        if k_i != 0:
            pos = pos_map.get((nums[index_i], index_i), -1)
            if pos != -1:
                count_tree.update(pos + 1, -1)
                sum_tree.update(pos + 1, -nums[index_i])

            low = 0
            high = n - 1
            answer_p = -1
            while low <= high:
                mid = (low + high) // 2
                current_count = count_tree.query(mid + 1)
                if current_count >= k_i:
                    answer_p = mid
                    high = mid - 1
                else:
                    low = mid + 1
            if answer_p != -1:
                sum_k = sum_tree.query(answer_p + 1)

        current_sum -= sum_k
        marked_sum += sum_k
        answers.append(current_sum)

    return answers