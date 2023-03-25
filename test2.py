cm = [[1, 2], [3, 4]]
cm_report = '\nConfusion Matrix:\n' + \
            '         |  Pred 0 |  Pred 1 |\n' + \
            '---------|---------|---------|\n' + \
            'Actual 0 |' + f'{cm[0][0]:^9}' + '|' + f'{cm[0][1]:^9}' + '|\n' + \
            'Actual 1 |' + f'{cm[1][0]:^9}' + '|' + f'{cm[1][1]:^9}' + '|\n'

print(cm_report)
