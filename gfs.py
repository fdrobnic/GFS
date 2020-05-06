import locale
import os
import tkinter as tk
import tkinter.filedialog as fdialog
from tkinter import font, messagebox, HORIZONTAL
from tkinter.ttk import Progressbar

import dill
import matplotlib as plt
from rfpimp import *
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV

import data
import disabled_cv
import display as dsp
from greedyfs import *

X_train = pd.DataFrame(columns=[0])
Xc = pd.DataFrame(columns=[0])
Y_train = pd.DataFrame(columns=[0])
SpearDF = pd.DataFrame(columns=[0])
mapping_ind = {}
w2 = ''
tuned_parameters = {}
est: GridSearchCV = object
est2: GridSearchCV = object
ld = []
tie_min_trees = True


def plot_hist():
    if len(Y_train):
        ax1 = dsp.plot_hist(Y_train=Y_train)
        ax1.set_title('Histogram of class values')
        ax1.get_figure().canvas.set_window_title('hist')
        plt.show()


def show_spearman():
    if len(X_train):
        global SpearDF
        ax1, m, SpearDF = dsp.show_spearman(X_train)
        r = 'Pairwise collinearity measure: %s' % m
        app.mText.insert('end-1c', '%s\n' % r)
        ax1.set_title('Spearman rank correlation coefficients (absolute value)')
        ax1.get_figure().canvas.set_window_title('spearman')
        plt.show()


def show_spearman_hm():
    if len(X_train):
        plot_corr_heatmap(X_train, figsize=(12, 10))
        plt.gca().set_aspect('auto')
        a = plt.gca()
        a.set_title('Spearman rank correlation coefficients heatmap')
        a.get_figure().canvas.set_window_title('corr')
        plt.show()


def show_feature_dep_matrix():
    if len(X_train):
        # Feature dependence matrix
        d = feature_dependence_matrix(X_train)
        pd.set_option('precision', 3)
        r = d['Dependence'].sort_values(ascending=False)
        app.mText.insert('end-1c', '%s\n' % r)
        viz = plot_dependence_heatmap(d, figsize=(11, 10))
        # viz.show()
        a = plt.gca()
        a.set_title('Feature dependence matrix')
        a.get_figure().canvas.set_window_title('corr_dep')
        plt.show()


def show_vif():
    if len(X_train):
        vdf = data.calc_vif(X_train)
        r = 'Multicollinearity measure (mean VIF): %s' % np.mean(vdf['value'])
        app.mText.insert('end-1c', '%s\n' % r)
        ax1 = vdf.plot(y='value', use_index=False, grid=True, logy=True)
        ax1.set_title('Variance inflation factors by feature')
        ax1.get_figure().canvas.set_window_title('vif')
        ax1.set_xlabel('No. of features')
        ax1.set_ylabel('VIF')
        plt.show()


def learn_base():
    global w2, tuned_parameters, X_train, Y_train, est
    if len(X_train):
        w = [2 ** a for a in range(0, int(np.floor(np.log2(len(X_train.columns)))) + 1)]
        w.append(len(X_train.columns))
        tuned_parameters = {'max_features': w,
                            'n_estimators': w2}
        mer_metoda = 'accuracy'
        # mer_metoda = None
        # kriterij = 'gini'
        kriterij = 'entropy'
        est = GridSearchCV(RandomForestClassifier(criterion=kriterij,
                                                  n_jobs=-1, oob_score=True,  # random_state=0,
                                                  verbose=False),
                           param_grid=tuned_parameters, cv=disabled_cv.DisabledCV,
                           scoring=mer_metoda, n_jobs=-1, error_score=np.nan)
        # prazen cv:
        # https://stackoverflow.com/questions/44636370/scikit-learn-gridsearchcv-without-cross-validation-unsupervised-learning

        # now est is the best classifier found given the search space
        est.fit(X_train, Y_train[0])
        r = '%s %s\n%s %s\n%s %s' % ('Best parameters:', est.best_params_,
                                     'Best OOB score:', est.best_estimator_.oob_score_,
                                     'Best CV score:', est.best_score_)
        app.mText.insert('end-1c', '%s\n' % r)


def importance_base():
    # correlation calculation
    Xc = X_train.copy()
    Xc['Y'] = Y_train[0]
    cc = Xc.corr()
    del Xc
    app.mText.insert('end-1c', 'Most important (built-in):\n')
    FI = est.best_estimator_.feature_importances_
    top_n = np.argsort(abs(FI))[:]
    app.mText.insert('end-1c', 'Feature;Importance;Correlation\n')
    f = []
    im = []
    for c in top_n:
        try:
            ccc = cc.loc[X_train.columns[c], 'Y']
        except KeyError:
            ccc = 0.
        app.mText.insert('end-1c', "%s;%s;%s\n" % (X_train.columns[c], locale.format("%g", FI[c]), locale.format("%g", ccc)))
        f.append(X_train.columns[c])
        im.append(FI[c])
    df = pd.DataFrame(data=im, columns=['importance'], index=f)
    ax = df.plot.barh()
    ax.set_title('Random Forests built-in feature importances')
    ax.set_xlabel('importance')
    ax.set_ylabel('feature')
    plt.show()


def predict_base():
    if len(X_train):
        predictions = est.predict(X_train)
        acc = np.mean(predictions == Y_train[0])

        primerjava = pd.DataFrame({'y': Y_train[0], 'predictions': predictions})
        r = 'Prediction results from base data:\n%s\n%s %s\n%s %s\n%s %s\n%s\n%s %s' % \
            (primerjava.describe(),
             'sum of abs. diffs:', sum(abs(primerjava['y'] - primerjava['predictions'])),
             'lowest diff:', min(primerjava['y'] - primerjava['predictions']),
             'highest diff:', max(primerjava['y'] - primerjava['predictions']),
             confusion_matrix(Y_train[0], predictions),
             'Final result:', acc)
        app.mText.insert('end-1c', '%s\n' % r)


def rfpimp_imp_base():
    if len(X_train):
        rfpimp_imp(est.best_estimator_, X_train, Y_train[0], 'pimp')


def rfpimp_imp_gfs():
    if len(X_train):
        rfpimp_imp(est2.best_estimator_, Xc, Y_train[0], 'pimp_2')


def rfpimp_imp(rf, x, y, naslov):
    if len(x):
        fig2 = plt.figure()
        imp = importances(rf, x, y)  # permutation
        r = 'Permutation feature importances:\n%s' % \
            '\n'.join(['%s %s' % (c[0], c[1].Importance) for c in imp.iterrows()])
        app.mText.insert('end-1c', '%s\n' % r)
        viz = plot_importances(imp)
        # viz.view()
        a = plt.gca()
        a.set_title('Permutation importances')
        a.get_figure().canvas.set_window_title(naslov)
        plt.show()


def rfpimp_dropc_base():
    if len(X_train):
        rfpimp_dropc(est.best_estimator_, X_train, Y_train[0], 'dcimp')


def rfpimp_dropc_gfs():
    if len(X_train):
        rfpimp_dropc(est2.best_estimator_, Xc, Y_train[0], 'dcimp_2')


def rfpimp_dropc(rf, x, y, naslov):
    if len(x):
        fig3 = plt.figure()
        imp = dropcol_importances(rf, x, y)  # drop columns
        r = 'Drop-column feature importances:\n%s' % '\n'.join(
                ['%s %s' % (c[0], c[1].Importance) for c in imp.iterrows()])
        app.mText.insert('end-1c', '%s\n' % r)
        viz = plot_importances(imp)
        # viz.view()
        a = plt.gca()
        a.set_title('Drop-column importances')
        a.get_figure().canvas.set_window_title(naslov)
        plt.show()


def show_acc_by_step():
    if len(ld):
        # plot of feature accuracies - by step:
        i = 0
        for l in ld:
            i += 1
            u1 = pd.DataFrame({'feature': pd.Series(list(l[3].keys())), 'acc': pd.Series(list(l[3].values()))}) \
                .sort_values(by='acc', axis=0, ascending=False)
            ax1 = u1.plot(x='feature', y='acc', rot=10)
            ax1.get_figure().canvas.set_window_title('kop_%s' % i)
        plt.show()


def show_growing_acc():
    if len(ld):
        app.mText.insert('end-1c', '%s\n' % 'Best performing features by step:')
        i = 0
        for l in ld:
            app.mText.insert('end-1c', 'step: %s feature: %s accuracy: %s\n' % (i, l[0], l[2]))
            for c in [c for c in l[3] if l[3][c] == l[2]]:
                app.mText.insert('end-1c', ' feature: %s parameters: %s\n' % (c, l[1][c]))
            i = i + 1

        # model performance plot by step:
        r = []
        for l in ld:
            r.append(l[2])
        rd = pd.DataFrame(r, columns=['score'])
        ax1 = rd.plot()
        ax1.get_figure().canvas.set_window_title('growth')
        plt.show()


def predict_reduced_ds():
    global est2
    if len(Xc):
        if not isinstance(est2, type(GridSearchCV)):
            est2 = clone(est)
            est2.estimator.set_params(ccp_alpha=0.0)
            est2.param_grid = adjust_tuned_par(Xc, tuned_parameters)
            est2.fit(Xc, Y_train[0])

        predictions = est2.best_estimator_.predict(Xc)
        acc = np.mean(predictions == Y_train[0])

        primerjava = pd.DataFrame({'y': Y_train[0], 'predictions': predictions})
        r = 'Prediction results after FS:\n%s\n%s %s\n%s %s\n%s %s\n%s\n%s %s' % \
            (primerjava.describe(),
             'sum of abs. diffs:', sum(abs(primerjava['y'] - primerjava['predictions'])),
             'lowest diff:', min(primerjava['y'] - primerjava['predictions']),
             'highest diff:', max(primerjava['y'] - primerjava['predictions']),
             confusion_matrix(Y_train[0], predictions),
             'Final result:', acc)
        app.mText.insert('end-1c', '%s\n' % r)


def show_growth_stat():
    axs = {}
    for t in [1, 2]:
        app.mText.insert('end-1c', 'type: %s\n' % t)
        r = []
        with os.scandir() as li:
            for entry in li:
                if entry.is_file() and ".pkl" in entry.name:
                    dill.load_session(entry.name)
                    s = 1
                    if (t == 1 and ld[0][0] in ['src_bytes', '\'src_bytes\''])\
                            or (t == 2 and ld[0][0] not in ['src_bytes', '\'src_bytes\'']):
                        app.mText.insert('end-1c', '%s\n' % entry.name)
                        for l in ld:
                            if ld[0][0][0] == '\'':
                                r.append([entry.name, s, l[1], l[0]])
                            elif isinstance(ld[0][2], type({})):
                                r.append([entry.name, s, l[2][l[0]], l[0]])
                            else:
                                r.append([entry.name, s, l[2], l[0]])
                            s += 1
        rd = pd.DataFrame(r, columns=['name', 'step', 'acc', 'feature'])
        ax: plt.Axes = rd.boxplot(by='step', column='acc')
        ax.get_figure().suptitle('')
        ax.set_title('Accuracy by step')
        ax.get_figure().canvas.set_window_title('growth')
        '''
        ax = rd.boxplot(by='step', column='acc')
        ax.get_figure().suptitle('')
        ax.set_title('Accuracy by step')
        ax.set_yscale('log')
        ax.get_figure().canvas.set_window_title('growth_log')
        '''
        axs[t] = ax

    xlim = axs[1].get_xlim()
    ylim = axs[1].get_ylim()
    axs[2].set_xlim(xlim[0], xlim[1])
    axs[2].set_ylim(ylim[0], ylim[1])
    plt.show()


def show_num_feat_stat():
    axs = {}
    ds = ['NSL-KDD', 'UCI-BCW']
    r = []
    for t in [1, 2]:
        app.mText.insert('end-1c', 'type: %s\n' % t)
        with os.scandir() as li:
            for entry in li:
                if entry.is_file() and ".pkl" in entry.name:
                    dill.load_session(entry.name)
                    s = 1
                    if (t == 1 and ld[0][0] in ['src_bytes', '\'src_bytes\''])\
                            or (t == 2 and ld[0][0] not in ['src_bytes', '\'src_bytes\'']):
                        app.mText.insert('end-1c', '%s\n' % entry.name)
                        for l in ld:
                            if ld[0][0][0] == '\'':
                                r.append([ds[t - 1], entry.name, s, l[1], l[0]])
                            elif isinstance(ld[0][2], type({})):
                                r.append([ds[t - 1], entry.name, s, l[2][l[0]], l[0]])
                            else:
                                r.append([ds[t - 1], entry.name, s, l[2], l[0]])
                            s += 1
    rd = pd.DataFrame(r, columns=['dataset', 'name', 'step', 'acc', 'feature'])
    rds = rd.groupby(by=['dataset', 'name']).agg('max')['step']
    rd2 = pd.DataFrame({'dataset': np.array(rds.index.get_level_values(0).values), 'step': np.array(rds.values)})
    ax: plt.Axes = rd2.boxplot(by='dataset', column='step')
    ax.get_figure().suptitle('')
    ax.set_title('Number of selected features')
    ax.get_figure().canvas.set_window_title('num_feat')

    plt.show()


def show_important_features_builtin():
    axs = {}
    global top_n, FI
    for t in [1, 2]:
        app.mText.insert('end-1c', 'type: %s\n' % t)
        r = []
        with os.scandir() as li:
            for entry in li:
                if entry.is_file() and ".pkl" in entry.name:
                    top_n = []
                    FI = []
                    dill.load_session(entry.name)
                    if (t == 1 and ld[0][0] in ['src_bytes', '\'src_bytes\''])\
                            or (t == 2 and ld[0][0] not in ['src_bytes', '\'src_bytes\'']):
                        app.mText.insert('end-1c', '%s\n' % entry.name)
                        if len(top_n):
                            app.mText.insert('end-1c', '%s;%s;%s;%s;%s;%s\n' % (X_train.columns[top_n[0]], locale.format("%g", FI[top_n[0]]),
                            X_train.columns[top_n[1]], locale.format("%g", FI[top_n[1]]),
                            X_train.columns[top_n[2]], locale.format("%g", FI[top_n[2]])))


class Application(tk.Tk):

    def __init__(self):
        tk.Tk.__init__(self)
        self.title('GFS')
        tk.Tk.protocol(self, "WM_DELETE_WINDOW", self.close_app)

        # Menubar

        menubar = tk.Menu(self)

        filemenu = tk.Menu(menubar, tearoff=0)
        filemenu.add_command(label="Load dill", command=self.load_dill)
        filemenu.add_command(label="Save state", command=self.save_state)
        filemenu.add_command(label="Load state", command=self.load_state)
        filemenu.add_separator()
        filemenu.add_command(label="Clear output", command=self.clear_output)
        filemenu.add_separator()
        filemenu.add_command(label="Exit", command=self.close_app)
        menubar.add_cascade(label="File", menu=filemenu)

        datamenu = tk.Menu(menubar, tearoff=0)
        datamenu.add_command(label="Provide UCI-BCW", command=self.provide_cancer)
        datamenu.add_command(label="Provide NSL-KDD", command=self.provide_KDD)
        datamenu.add_separator()
        datamenu.add_command(label="Load last UCI-BCW", command=self.load_data_cancer)
        datamenu.add_command(label="Load last NSL-KDD", command=self.load_data_KDD)
        menubar.add_cascade(label="Data", menu=datamenu)

        descmenu = tk.Menu(menubar, tearoff=0)
        descmenu.add_command(label="Histogram", command=plot_hist)
        descmenu.add_command(label="Multicollinearity", command=show_spearman)
        descmenu.add_command(label="Spearman heatmap", command=show_spearman_hm)
        descmenu.add_command(label="Feature dep. matrix", command=show_feature_dep_matrix)
        descmenu.add_separator()
        descmenu.add_command(label="VIF", command=show_vif)
        menubar.add_cascade(label="Descriptive", menu=descmenu)

        baselmenu = tk.Menu(menubar, tearoff=0)
        baselmenu.add_command(label="Learn", command=learn_base)
        baselmenu.add_command(label="Importance", command=importance_base)
        baselmenu.add_command(label="Predict", command=predict_base)
        baselmenu.add_separator()
        baselmenu.add_command(label="Importance stat.", command=show_important_features_builtin)
        baselmenu.add_separator()
        baselmenu.add_command(label="Permut. imp.", command=rfpimp_imp_base)
        baselmenu.add_command(label="Drop-column imp.", command=rfpimp_dropc_base)
        menubar.add_cascade(label="Base learning", menu=baselmenu)

        gfs_menu = tk.Menu(menubar, tearoff=0)
        gfs_menu.add_command(label="Calculate", command=self.show_calc_gfs)
        gfs_menu.add_command(label="Predict", command=predict_reduced_ds)
        gfs_menu.add_separator()
        gfs_menu.add_command(label="Cumulative feature imp.", command=show_acc_by_step)
        gfs_menu.add_command(label="Accuracy growth", command=show_growing_acc)
        gfs_menu.add_command(label="Show growth stat.", command=show_growth_stat)
        gfs_menu.add_command(label="Show num. sel. feat.", command=show_num_feat_stat)
        gfs_menu.add_separator()
        gfs_menu.add_command(label="Permut. imp.", command=rfpimp_imp_gfs)
        gfs_menu.add_command(label="Drop-column imp.", command=rfpimp_dropc_gfs)
        menubar.add_cascade(label="Greedy FS", menu=gfs_menu)

        self.config(menu=menubar)

        self.dataset_t = tk.Label(text='Dataset name:')
        self.dataset_t.grid(row=0, column=0)
        self.dataset_name = ''
        self.dataset = tk.Label(text=self.dataset_name)
        self.dataset.grid(row=0, column=1, sticky='NSEW')

        # Text widget, its font and frame

        self.defaultFont = font.Font(name="defFont")
        d = self.defaultFont.actual()

        textFrame = tk.Frame(self, borderwidth=1, relief="sunken", width=600, height=600)

        textFrame.grid_propagate(False)  # ensures a consistent GUI size
        textFrame.grid(row=1, columnspan=2, sticky='NSEW')

        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

        textFrame.columnconfigure(0, weight=1)
        textFrame.rowconfigure(0, weight=1)

        self.mText = tk.Text(textFrame, width=48, height=1, wrap='word', font="TkFixedFont")
        self.mText.grid(row=0, column=0, sticky='NSEW')

        # Scrollbar and config

        tScrollbar = tk.Scrollbar(textFrame, command=self.mText.yview)
        tScrollbar.grid(row=0, column=1, sticky='NSEW', pady=1)

        self.mText.config(yscrollcommand=tScrollbar.set)

        # Stretchable

        textFrame.grid_rowconfigure(0, weight=1)
        textFrame.grid_columnconfigure(0, weight=1)

        # Center main window

        self.update_idletasks()

        xp = int((self.winfo_screenwidth() / 2) - (self.winfo_width() / 2) - 8)
        yp = int((self.winfo_screenheight() / 2) - (self.winfo_height() / 2) - 30)
        self.geometry('{0}x{1}+{2}+{3}'.format(self.winfo_width(), self.winfo_height(),
                                               xp, yp))

        # train parameters window
        self.paramwin = tk.Toplevel(self, bd=4)  # , relief='ridge')
        l_w = tk.Label(self.paramwin, text='Param max_features:')
        l_w.grid(row=0, column=0)
        self.w_text = tk.Entry(self.paramwin, width=60)
        self.w_text.grid(row=0, column=1)
        l_w2 = tk.Label(self.paramwin, text='Param n_estimators:')
        l_w2.grid(row=1, column=0)
        self.w2_text = tk.Entry(self.paramwin, width=60)
        self.w2_text.grid(row=1, column=1)

        p_start_button = tk.Button(self.paramwin, text='Start', command=self.calc_gfs)
        p_start_button.grid(row=2, column=0)

        p_cancel_button = tk.Button(self.paramwin, text='Cancel', command=self.paramwin.withdraw)
        p_cancel_button.grid(row=2, column=1)

        l_s = tk.Label(self.paramwin, text='Step:')
        l_s.grid(row=3, column=0)
        self.s_text = tk.Entry(self.paramwin, width=60)
        self.s_text.grid(row=3, column=1)

        self.progress = Progressbar(self.paramwin, orient=HORIZONTAL, length=200, mode='determinate')
        self.progress.grid(row=4, columnspan=2, sticky='NSEW')

        self.paramwin.overrideredirect(True)  # No outerframe!

        self.paramwin.withdraw()

        # Bindings
        self.bind_class("Text", "<Control-a>", self.select_all)

    def select_all(self, event):
        self.mText.tag_add("sel", "1.0", "end-1c")

    def clear_output(self):
        self.mText.delete('1.0', 'end-1c')

    def close_app(self):
        plt.close('all')
        self.quit()
        self.destroy()

    def set_dataset_name(self, name):
        self.dataset_name = name
        self.dataset.configure(text=self.dataset_name)

    def provide_cancer(self):
        global X_train
        global Y_train
        global mapping_ind
        global w2
        X_train, Y_train, mapping_ind = data.provide_cancer()
        w2 = [a ** 2 for a in range(1, 10)]  # for UCI-BCW
        # messagebox.showinfo("Success", "Data imported (%s rows)" % len(X_train))
        self.set_dataset_name('UCI-BCW')

    def provide_KDD(self):
        global X_train
        global Y_train
        global mapping_ind
        global w2
        X_train, Y_train, mapping_ind = data.provide_KDD()
        w2 = [a ** 2 for a in range(4, 12)]  # for NSL-KDD
        # messagebox.showinfo("Success", "Data imported (%s rows)" % len(X_train))
        self.set_dataset_name('NSL-KDD')

    def load_data_cancer(self):
        global X_train
        global Y_train
        global mapping_ind
        global w2
        data.load_data_cancer()
        w2 = [a ** 2 for a in range(1, 10)]  # for UCI-BCW
        # messagebox.showinfo("Success", "Data imported (%s rows)" % len(X_train))
        self.set_dataset_name('UCI-BCW')

    def load_data_KDD(self):
        global X_train
        global Y_train
        global mapping_ind
        global w2
        data.load_data_KDD()
        w2 = [a ** 2 for a in range(4, 12)]  # for NSL-KDD
        # messagebox.showinfo("Success", "Data imported (%s rows)" % len(X_train))
        self.set_dataset_name('NSL-KDD')

    def load_dill(self):
        fname = fdialog.askopenfilename(filetypes=['dill {.pkl}'], title='Open dill PKL file', defaultextension='PKL')
        if fname:
            dill.load_session(fname)
            self.set_dataset_name(fname)
            messagebox.showinfo("Success", "Data loaded.")

    def save_state(self):
        data_ = {}
        for obj in ['X_train',
                    'Xc',
                    'Y_train',
                    'SpearDF',
                    'mapping_ind',
                    'w2',
                    'tuned_parameters',
                    'est',
                    'est2',
                    'ld',
                    'tie_min_trees']:
            print(obj, str(type(eval(obj))))
            data_[obj] = eval(obj)
        with open('state.pkl', 'wb') as f:
            dill.dump(data_, f)

        messagebox.showinfo("Success", "Data saved.")

    def load_state(self):
        global X_train, Xc, Y_train, SpearDF, mapping_ind, w2, tuned_parameters, est, est2, ld, tie_min_trees
        fname = 'state.pkl'
        try:
            with open(fname, 'rb') as f:
                data_ = dill.load(f, ignore=True)
        except FileNotFoundError:
            messagebox.showerror("Error", "File %s not found." % fname)
            return None
        X_train = data_['X_train']
        Xc = data_['Xc']
        Y_train = data_['Y_train']
        SpearDF = data_['SpearDF']
        mapping_ind = data_['mapping_ind']
        w2 = data_['w2']
        tuned_parameters = data_['tuned_parameters']
        est = data_['est']
        est2 = data_['est2']
        ld = data_['ld']
        tie_min_trees = data_['tie_min_trees']

        messagebox.showinfo("Success", "Data loaded.")

    def show_calc_gfs(self):
        global w, w2
        if len(X_train):
            if len(self.w_text.get()) > 0:
                self.w_text.delete(0, 'end')
            self.w_text.insert('end', str(w))
            if len(self.w2_text.get()) > 0:
                self.w2_text.delete(0, 'end')
            self.w2_text.insert('end', str(w2))
            self.paramwin.deiconify()
            xpos = self.winfo_rootx() + self.winfo_width() + 8
            ypos = self.winfo_rooty()
            self.paramwin.geometry('{0}x{1}+{2}+{3}'.format(self.paramwin.winfo_width(),
                                                            self.paramwin.winfo_height(), xpos, ypos))

    def calc_gfs(self):
        global w, w2, tuned_parameters, Xc, ld, est2
        w = eval(self.w_text.get())
        w2 = eval(self.w2_text.get())
        tuned_parameters = {'max_features': w,
                            'n_estimators': w2}
        mn = 0.5  # število dovoljenih napačnih sklepanj
        margin = (1 - (mn / len(X_train))) * est.best_score_
        Xc, ld = greedy_feature_selection(est, X_train, Y_train, margin, tuned_parameters,
                                          self.refresh_progress, tie_min_trees)
        est2 = clone(est)
        est2.param_grid = adjust_tuned_par(Xc, tuned_parameters)
        est2.fit(Xc, Y_train[0])

        self.paramwin.withdraw()

    def refresh_progress(self, s, n, m):
        if len(self.s_text.get()) > 0:
            self.s_text.delete(0, 'end')
        self.s_text.insert('end', str(s))
        self.progress["value"] = n
        self.progress["maximum"] = m
        self.update_idletasks()


if __name__ == '__main__':
    plt.rcParams.update({'figure.max_open_warning': 100})
    locale.setlocale(locale.LC_ALL, "")

    app = Application()
    app.mainloop()
    app.quit()
