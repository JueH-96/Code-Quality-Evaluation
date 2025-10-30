import itertools

class Solution:
    def buyChoco(self, prices: List[int], money: int) -> int:
        pairs = itertools.combinations(prices, 2)
        possible_sums = []
        for pair in pairs:
            s = sum(pair)
            if s <= money:
                possible_sums.append(s)
        if not possible_sums:
            return money
        else:
            min_sum = min(possible_sums)
            return money - min_sum