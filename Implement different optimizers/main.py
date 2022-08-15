from model import Regressor
import seaborn as sns
import matplotlib.pyplot as plt


if __name__ == '__main__':
    iters = 500
    loss = {}

    """gradient descent"""
    model1 = Regressor()
    loss['gd'] = model1.fit('gd', alpha=0.006, n_iters=iters)

    """sgd"""
    model2 = Regressor()
    loss['sgd'] = model2.fit('sgd', n_iters=iters, alpha=0.15, batch_size=20)

    """momentum sgd"""
    model3 = Regressor()
    loss['sgdm'] = model3.fit('sgdm', n_iters=iters,
                              momentum=0.99, alpha=0.15, batch_size=5)

    """adagrad"""
    model4 = Regressor()
    loss['adagrad'] = model4.fit(
        'adagrad', n_iters=iters, g=0, epsilon=5, render_animation=False)

    """rmsprop"""
    model5 = Regressor()
    loss['rmsprop'] = model5.fit(
        'rmsprop', n_iters=iters, g=0, alpha=0.99, epsilon=0.3)

    """adam"""
    model6 = Regressor()
    loss['adam'] = model6.fit('adam', n_iters=iters,
                              m=0, v=0, beta1=0.001, beta2=0.8)

    """Draw line plot to compare all loss diminish"""
    for m in loss.keys():
        while len(loss[m]) < 500:
            loss[m].append(loss[m][-1])
        sns.lineplot(x=[i for i in range(500)], y=loss[m], label=m)

    plt.ylabel('loss')
    plt.xlabel('iteration')
    plt.savefig('./figs/compare_all.png')
    plt.close()
    plt.show()