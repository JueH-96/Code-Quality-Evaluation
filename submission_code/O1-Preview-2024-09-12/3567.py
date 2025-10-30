class Solution:
    def convertDateToBinary(self, date: str) -> str:
        y_str, m_str, d_str = date.split('-')
        y = int(y_str)
        m = int(m_str)
        d = int(d_str)
        y_bin = bin(y)[2:]
        m_bin = bin(m)[2:]
        d_bin = bin(d)[2:]
        return f'{y_bin}-{m_bin}-{d_bin}'