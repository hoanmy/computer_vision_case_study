# computer_vision_case_study

Nhận xét:

1/ Các mô hình đều cho kết quả thấp, chỉ có KNeighborsClassifier với k = 3 thì cho kết quả tạm được.

2/ Số lượng hình dùng train quá ít, chất lượng thấp

3/ Các thông số cần tinh chỉnh hơn cho phù hợp.


-------------------------------***************----------------------------------

+++   KNeighborsClassifier with k = 3   +++

Model path: trained_Nearest_Neighbors_model.clf
KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
                     metric_params=None, n_jobs=None, n_neighbors=3, p=2,
                     weights='uniform')

Looking for faces in IMG_3.jpg
closest_distances:
[0.         0.         0.66666667 0.         0.         0.
 0.         0.         0.         0.33333333 0.        ]
- Found MINHHA at (3593, 1953)
- Found VULQ at (1632, 1846)
- Found unknown at (290, 1574)
- Found TULG at (999, 1056)
- Found VULQ at (1253, 1171)
- Found VULQ at (1716, 1035)
- Found TANTD at (2454, 1010)
- Found QUANVM at (1852, 1130)
- Found TANTD at (1876, 1455)
- Found LOCTH at (2941, 1381)
- Found HOAINT at (1150, 1439)

=> RESULT: 65%

Looking for faces in IMG_2.png
closest_distances:
[0.         0.         0.33333333 0.         0.         0.
 0.         0.         0.         0.         0.         0.
 0.         0.         0.33333333 0.        ]
- Found MINHHA at (892, 554)
- Found MINHHA at (97, 1006)
- Found DUYNN at (1409, 683)
- Found VULQ at (585, 611)
- Found QUANVM at (1934, 985)
- Found QUANVM at (2112, 586)
- Found HOAINT at (2269, 636)
- Found VULQ at (1465, 578)
- Found QUANVM at (1107, 749)
- Found QUANVM at (414, 605)
- Found TULG at (1787, 516)
- Found HOAISD at (1001, 634)
- Found TULG at (1149, 503)
- Found VULQ at (2319, 536)
- Found DUYNN at (2043, 550)
- Found DUYETLV at (840, 523)

=> RESULT: 45%

Looking for faces in IMG_1.png
closest_distances:
[0.         0.         0.33333333 0.         0.         0.         0.        ]
- Found VULQ at (153, 583)
- Found VULQ at (1005, 944)
- Found DUYNN at (855, 752)
- Found DUYETLV at (626, 626)
- Found HOAISD at (1583, 841)
- Found TULG at (1085, 555)
- Found VULQ at (1584, 677)

=> RESULT: 50%


-------------------------------***************----------------------------------

+++    GaussianProcessClassifier    +++

Model path: trained_Gaussian_Process_model.clf
GaussianProcessClassifier(copy_X_train=True, kernel=1**2 * RBF(length_scale=1),
                          max_iter_predict=100, multi_class='one_vs_rest',
                          n_jobs=None, n_restarts_optimizer=0,
                          optimizer='fmin_l_bfgs_b', random_state=None,
                          warm_start=False)

Looking for faces in IMG_3.jpg
closest_distances:
[0.07650153 0.07955944 0.246244   0.08492794 0.0531397  0.08474978
 0.07252074 0.04615898 0.06346259 0.08337886 0.0548477 ]
- Found MINHHA at (3593, 1953)
- Found VULQ at (1632, 1846)
- Found DUYNN at (290, 1574)
- Found TULG at (999, 1056)
- Found MINHHA at (1253, 1171)
- Found VULQ at (1716, 1035)
- Found MINHHA at (2454, 1010)
- Found QUANVM at (1852, 1130)
- Found TANTD at (1876, 1455)
- Found LOCTH at (2941, 1381)
- Found HOAINT at (1150, 1439)

=> RESULT: 65%

Looking for faces in IMG_2.png
closest_distances:
[0.05414611 0.06891335 0.06815074 0.08193586 0.06054978 0.05185012
 0.05774526 0.0559654  0.05145373 0.04669898 0.06741439 0.06997507
 0.06975457 0.06181846 0.09005801 0.06463312]
- Found MINHHA at (892, 554)
- Found TANTD at (97, 1006)
- Found HOAISD at (1409, 683)
- Found VULQ at (585, 611)
- Found TAIHPT at (1934, 985)
- Found QUANVM at (2112, 586)
- Found HOAINT at (2269, 636)
- Found VULQ at (1465, 578)
- Found LOCTH at (1107, 749)
- Found QUANVM at (414, 605)
- Found TULG at (1787, 516)
- Found MINHHA at (1001, 634)
- Found TULG at (1149, 503)
- Found VULQ at (2319, 536)
- Found TANTD at (2043, 550)
- Found DUYETLV at (840, 523)

=> RESULT: 65%

Looking for faces in IMG_1.png
closest_distances:
[0.07086339 0.04542778 0.07140003 0.07549129 0.06150728 0.06764018
 0.05213968]
- Found VULQ at (153, 583)
- Found VULQ at (1005, 944)
- Found LOCTH at (855, 752)
- Found DUYETLV at (626, 626)
- Found HOAISD at (1583, 841)
- Found TULG at (1085, 555)
- Found VULQ at (1584, 677)

=> RESULT: 55%


-------------------------------***************----------------------------------

+++   DecisionTreeClassifier  +++

Model path: trained_Decision_Tree_model.clf
DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=5,
                       max_features=None, max_leaf_nodes=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, presort=False,
                       random_state=None, splitter='best')

Looking for faces in IMG_3.jpg
closest_distances:
[0.         0.11111111 0.11111111 0.11111111 0.11111111 0.11111111
 0.11111111 0.11111111 0.11111111 0.         0.11111111]
- Found MINHHA at (3593, 1953)
- Found DUYNN at (1632, 1846)
- Found DUYNN at (290, 1574)
- Found DUYNN at (999, 1056)
- Found DUYNN at (1253, 1171)
- Found DUYNN at (1716, 1035)
- Found DUYNN at (2454, 1010)
- Found DUYNN at (1852, 1130)
- Found DUYNN at (1876, 1455)
- Found VULQ at (2941, 1381)
- Found DUYNN at (1150, 1439)

=> RESULT: 15%

Looking for faces in IMG_2.png
closest_distances:
[0.         0.11111111 0.11111111 0.11111111 0.11111111 0.11111111
 0.11111111 0.         0.         0.11111111 0.11111111 0.
 0.11111111 0.         0.11111111 0.11111111]
- Found MINHHA at (892, 554)
- Found DUYNN at (97, 1006)
- Found DUYNN at (1409, 683)
- Found DUYNN at (585, 611)
- Found DUYNN at (1934, 985)
- Found DUYNN at (2112, 586)
- Found DUYNN at (2269, 636)
- Found VULQ at (1465, 578)
- Found LOCTH at (1107, 749)
- Found DUYNN at (414, 605)
- Found DUYNN at (1787, 516)
- Found LOCTH at (1001, 634)
- Found DUYNN at (1149, 503)
- Found MINHHA at (2319, 536)
- Found DUYNN at (2043, 550)
- Found DUYNN at (840, 523)

=> RESULT: 15%

Looking for faces in IMG_1.png
closest_distances:
[0.11111111 0.         0.         0.11111111 0.11111111 0.11111111
 0.        ]
- Found DUYNN at (153, 583)
- Found LOCTH at (1005, 944)
- Found LOCTH at (855, 752)
- Found DUYNN at (626, 626)
- Found DUYNN at (1583, 841)
- Found DUYNN at (1085, 555)
- Found MINHHA at (1584, 677)

=> RESULT: 10%


-------------------------------***************----------------------------------

+++  RandomForestClassifier   +++

Model path: trained_Random_Forest_model.clf
RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                       max_depth=5, max_features=1, max_leaf_nodes=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, n_estimators=10,
                       n_jobs=None, oob_score=False, random_state=None,
                       verbose=0, warm_start=False)


Looking for faces in IMG_3.jpg
closest_distances:
[0.         0.10666667 0.3        0.04       0.         0.
 0.         0.         0.         0.10666667 0.        ]
- Found MINHHA at (3593, 1953)
- Found TRUONGTX at (1632, 1846)
- Found DUYNN at (290, 1574)
- Found TIENBDT at (999, 1056)
- Found DUYETLV at (1253, 1171)
- Found LOCTH at (1716, 1035)
- Found LOCTH at (2454, 1010)
- Found HOAINT at (1852, 1130)
- Found MINHHA at (1876, 1455)
- Found LOCTH at (2941, 1381)
- Found HOAINT at (1150, 1439)

=> RESULT: 50%

Looking for faces in IMG_2.png
closest_distances:
[0.         0.         0.         0.         0.         0.
 0.         0.04       0.         0.         0.06666667 0.
 0.10666667 0.04       0.1        0.        ]
- Found QUANVM at (892, 554)
- Found HANTQ at (97, 1006)
- Found HOAISD at (1409, 683)
- Found MINHHA at (585, 611)
- Found TULG at (1934, 985)
- Found LOCTH at (2112, 586)
- Found HOAINT at (2269, 636)
- Found LOCTH at (1465, 578)
- Found QUANVM at (1107, 749)
- Found QUANVM at (414, 605)
- Found TANTD at (1787, 516)
- Found MINHHA at (1001, 634)
- Found TIENBDT at (1149, 503)
- Found TULG at (2319, 536)
- Found MYNH at (2043, 550)
- Found QUANVM at (840, 523)

=> RESULT: 65%

Looking for faces in IMG_1.png
closest_distances:
[0.         0.04       0.         0.         0.         0.10666667
 0.04      ]
- Found DUYETLV at (153, 583)
- Found LOCTH at (1005, 944)
- Found LOCTH at (855, 752)
- Found QUANVM at (626, 626)
- Found HUNGNV at (1583, 841)
- Found QUANVM at (1085, 555)
- Found MINHHA at (1584, 677)

=> RESULT: 10%


-------------------------------***************----------------------------------

+++  MLPClassifier(sklearn.neural_network)  +++

Model path: trained_Neural_Net_model.clf
MLPClassifier(activation='relu', alpha=1, batch_size='auto', beta_1=0.9,
              beta_2=0.999, early_stopping=False, epsilon=1e-08,
              hidden_layer_sizes=(100,), learning_rate='constant',
              learning_rate_init=0.001, max_iter=1000, momentum=0.9,
              n_iter_no_change=10, nesterovs_momentum=True, power_t=0.5,
              random_state=None, shuffle=True, solver='adam', tol=0.0001,
              validation_fraction=0.1, verbose=False, warm_start=False)



Looking for faces in IMG_3.jpg
closest_distances:
[0.05037924 0.07274287 0.0744342  0.06350519 0.06042727 0.06805042
 0.05065309 0.05621602 0.03371758 0.07441964 0.04611359]
- Found MINHHA at (3593, 1953)
- Found VULQ at (1632, 1846)
- Found VULQ at (290, 1574)
- Found MINHHA at (999, 1056)
- Found MINHHA at (1253, 1171)
- Found MINHHA at (1716, 1035)
- Found MINHHA at (2454, 1010)
- Found MINHHA at (1852, 1130)
- Found TANTD at (1876, 1455)
- Found DUYETLV at (2941, 1381)
- Found TANTD at (1150, 1439)

=> RESULT: 10%

Looking for faces in IMG_2.png
closest_distances:
[0.05058558 0.05045535 0.06553223 0.06111575 0.06155503 0.05944709
 0.04178598 0.06093894 0.06046731 0.05859567 0.06398787 0.05428259
 0.06464265 0.0545816  0.06068082 0.0548578 ]
- Found DUYETLV at (892, 554)
- Found MINHHA at (97, 1006)
- Found MINHHA at (1409, 683)
- Found MINHHA at (585, 611)
- Found MINHHA at (1934, 985)
- Found MINHHA at (2112, 586)
- Found TANTD at (2269, 636)
- Found MINHHA at (1465, 578)
- Found DUYETLV at (1107, 749)
- Found MINHHA at (414, 605)
- Found MINHHA at (1787, 516)
- Found MINHHA at (1001, 634)
- Found MINHHA at (1149, 503)
- Found MINHHA at (2319, 536)
- Found MINHHA at (2043, 550)
- Found DUYETLV at (840, 523)

=> RESULT: 10%


Looking for faces in IMG_1.png
closest_distances:
[0.06178256 0.0630782  0.05630192 0.05441834 0.05846283 0.06745062
 0.06219261]
- Found MINHHA at (153, 583)
- Found DUYETLV at (1005, 944)
- Found MINHHA at (855, 752)
- Found DUYETLV at (626, 626)
- Found MINHHA at (1583, 841)
- Found MINHHA at (1085, 555)
- Found MINHHA at (1584, 677)

=> RESULT: 0%


-------------------------------***************----------------------------------

+++  AdaBoostClassifier(https://scikit-learn.org/stable/modules/ensemble.html#adaboost)   +++

Model path: trained_AdaBoost_model.clf
AdaBoostClassifier(algorithm='SAMME.R', base_estimator=None, learning_rate=1.0,
                   n_estimators=50, random_state=None)


Looking for faces in IMG_3.jpg
closest_distances:
[0.0491926  0.04236103 0.04947678 0.0462109  0.04947678 0.04947678
 0.04794278 0.04541305 0.04654743 0.03302966 0.04922161]
- Found MINHHA at (3593, 1953)
- Found MINHHA at (1632, 1846)
- Found MINHHA at (290, 1574)
- Found MINHHA at (999, 1056)
- Found MINHHA at (1253, 1171)
- Found MINHHA at (1716, 1035)
- Found MINHHA at (2454, 1010)
- Found MINHHA at (1852, 1130)
- Found MINHHA at (1876, 1455)
- Found TULG at (2941, 1381)
- Found MINHHA at (1150, 1439)

=> RESULT: 0%

Looking for faces in IMG_2.png
closest_distances:
[0.04891591 0.07157758 0.04302525 0.04947678 0.03506685 0.03712993
 0.03706867 0.04947678 0.04947678 0.04119589 0.04133084 0.04929018
 0.04947678 0.04918982 0.02924021 0.04947678]
- Found MINHHA at (892, 554)
- Found HANTQ at (97, 1006)
- Found HOAISD at (1409, 683)
- Found MINHHA at (585, 611)
- Found TAIHPT at (1934, 985)
- Found QUANVM at (2112, 586)
- Found HOAINT at (2269, 636)
- Found MINHHA at (1465, 578)
- Found MINHHA at (1107, 749)
- Found QUANVM at (414, 605)
- Found QUANVM at (1787, 516)
- Found MINHHA at (1001, 634)
- Found MINHHA at (1149, 503)
- Found MINHHA at (2319, 536)
- Found QUANVM at (2043, 550)
- Found MINHHA at (840, 523)

=> RESULT: 20%

Looking for faces in IMG_1.png
closest_distances:
[0.04289275 0.04944958 0.04302525 0.03674605 0.04265076 0.04922161
 0.04947678]
- Found NHANTH at (153, 583)
- Found MINHHA at (1005, 944)
- Found HOAISD at (855, 752)
- Found QUANVM at (626, 626)
- Found HOAISD at (1583, 841)
- Found MINHHA at (1085, 555)
- Found MINHHA at (1584, 677)

=> RESULT: 5%


-------------------------------***************----------------------------------

+++   GaussianNB  +++

Model path: trained_Naive_Bayes_model.clf
GaussianNB(priors=None, var_smoothing=1e-09)

Looking for faces in IMG_3.jpg
closest_distances:
[0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
- Found MINHHA at (3593, 1953)
- Found MINHHA at (1632, 1846)
- Found MINHHA at (290, 1574)
- Found MINHHA at (999, 1056)
- Found MINHHA at (1253, 1171)
- Found MINHHA at (1716, 1035)
- Found MINHHA at (2454, 1010)
- Found MINHHA at (1852, 1130)
- Found MINHHA at (1876, 1455)
- Found MINHHA at (2941, 1381)
- Found MINHHA at (1150, 1439)

=> RESULT: 0%

Looking for faces in IMG_2.png
closest_distances:
[0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
- Found MINHHA at (892, 554)
- Found MINHHA at (97, 1006)
- Found MINHHA at (1409, 683)
- Found MINHHA at (585, 611)
- Found MINHHA at (1934, 985)
- Found MINHHA at (2112, 586)
- Found MINHHA at (2269, 636)
- Found MINHHA at (1465, 578)
- Found MINHHA at (1107, 749)
- Found MINHHA at (414, 605)
- Found MINHHA at (1787, 516)
- Found MINHHA at (1001, 634)
- Found MINHHA at (1149, 503)
- Found MINHHA at (2319, 536)
- Found MINHHA at (2043, 550)
- Found MINHHA at (840, 523)

=> RESULT: 0%

Looking for faces in IMG_1.png
closest_distances:
[0. 0. 0. 0. 0. 0. 0.]
- Found MINHHA at (153, 583)
- Found MINHHA at (1005, 944)
- Found MINHHA at (855, 752)
- Found MINHHA at (626, 626)
- Found MINHHA at (1583, 841)
- Found MINHHA at (1085, 555)
- Found MINHHA at (1584, 677)

=> RESULT: 0%
