# -*- coding: utf-8 -*-
"""
07_drawfig.py
--------------------------------------------------------------------------------
Generates result figures from training .pkl files.
--------------------------------------------------------------------------------
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import MultipleLocator
import config

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
prepath  = config.RESULTS_SUMMER_DIR + "/"
prepathw = config.RESULTS_WINTER_DIR + "/"
OUTDIR   = SCRIPT_DIR
os.makedirs(OUTDIR, exist_ok=True)

SAVE_EXTS = [".png"]


def _savefig(savedir, savename, nums):
    """Unified save: export in all SAVE_EXTS formats. Skip if savedir is empty."""
    if savedir == "":
        plt.close()
        return
    name = savename if savename is not None else str(nums)
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    for ext in SAVE_EXTS:
        plt.savefig(os.path.join(savedir, name + ext), bbox_inches='tight')
    plt.close()


def get_basename(filename):
    return os.path.basename(filename)


def get_allfiles(path):
    all_files = []
    file_list = os.listdir(path)
    for file in file_list:
        cur_path = os.path.join(path, file)
        if os.path.isdir(cur_path):
            all_files.extend(get_allfiles(cur_path))
        else:
            all_files.append(path + "/" + file)
    return all_files


def get_alldirs(path):
    all_files = []
    file_list = os.listdir(path)
    for file in file_list:
        cur_path = os.path.join(path, file)
        all_files.append(path + "/" + file)
        if os.path.isdir(cur_path):
            all_files.extend(get_alldirs(cur_path))
    return all_files


def get_filesbyend(filelist, endstr=".pkl"):
    files = []
    for a in filelist:
        try:
            if a.endswith(endstr):
                files.append(a)
        except Exception:
            continue
    return files


def get_FName(stro):
    return os.path.basename(stro)


def get_FNameBySplit(stro, pos=0, str="_", num=-1):
    strs = os.path.basename(stro).split(str, num)
    if len(strs) >= pos + 1:
        return strs[pos]
    else:
        return ""


def strtointarray(str):
    c = []
    for i in str.split(','):
        c.append(int(i.strip().strip('[]')))
    return c


# Each record: [mnum, wt, day, p, epoch, acc, recall, r0, r1, r2, r3, r4]
def build_result(prepath):
    files = get_filesbyend(get_alldirs(prepath))
    files.sort()
    listn = [get_FName(i) for i in files]
    result = {}
    for i in listn:
        info   = get_FNameBySplit(i, 2, "_")
        tm     = get_FNameBySplit(i, 0, "_")
        tmnum  = get_FNameBySplit(tm, 1, "-")
        tmn    = get_FNameBySplit(tm, 0, "-")
        tr     = get_FNameBySplit(i, 4, "_")
        tr0    = get_FNameBySplit(i, 11, "_")
        tr1    = get_FNameBySplit(i, 12, "_")
        tr2    = get_FNameBySplit(i, 13, "_")
        tr3    = get_FNameBySplit(i, 14, "_")
        tr4    = get_FNameBySplit(i, 15, "_").replace(".pkl", "")
        ta     = get_FNameBySplit(i, 6, "_")
        twt    = get_FNameBySplit(i, 1, "_")
        td     = get_FNameBySplit(info, 1, "-")
        tp     = get_FNameBySplit(info, 3, "-")
        tep    = get_FNameBySplit(i, 8, "_")
        try:
            row = [int(tmnum), int(twt), int(td), float(tp), int(tep),
                   float(ta), float(tr),
                   float(tr0), float(tr1), float(tr2), float(tr3), float(tr4)]
        except ValueError:
            continue
        result.setdefault(tmn, []).append(row)
    return result


_INDEXES = {}
def _index_of(result):
    rid = id(result)
    if rid not in _INDEXES:
        idx = {}
        for mn, rows in result.items():
            sub = {}
            for r in rows:
                sub.setdefault((r[0], r[1], r[2], r[3]), []).append(r[4:12])
            for k, v in sub.items():
                sub[k] = np.array(v) if len(v) else np.empty((0, 8))
            idx[mn] = sub
        _INDEXES[rid] = idx
    return _INDEXES[rid]


def getdata(mn, mnum, wt, d, p, result):
    """Return [epoch, acc, recall, r0, r1, r2, r3, r4] arrays for matching records."""
    sub = _index_of(result).get(mn, {})
    arr = sub.get((mnum, wt, d, p), None)
    return arr if arr is not None else np.empty((0, 8))


def getdataarray(mname, nl=[0, 1, 2], wtl=[1, 2], dl=[1, 2, 3, 4, 5],
                 pl=[1, 0.75, 0.5, 0.25], col=2, result=None):
    if result is None:
        result = RESULT_SUMMER
    rd = []
    for n in nl:
        for wt in wtl:
            for d in dl:
                for p in pl:
                    data = getdata(mname, n, wt, d, p, result)
                    if len(data) >= 1:
                        data = data[data[:, 2].argsort()][::-1]
                        rd.append(data[0][col])
                    else:
                        rd.append(np.nan)
    return rd


def drawfig(nums=range(0, 100),
            resultdata=None,
            reshapedata=(-1, 5),
            datakey=["transformer", "Capsule", "resnet", "CNN", "LogisticRegression"],
            x=["day-1", "day-2", "day-3", "day-4", "day-5"],
            color=["red", "green", "blue", "orange", "black"],
            wtl=[2],
            days=[1, 2, 3, 4, 5],
            p=[1],
            col=[1, 2],
            fill_between=[False, False],
            figsize=(10, 6.18), dpi=100,
            haslegend=True,
            legendloc="upper right",
            ncol=1,
            legendarr=[],
            title="",
            xlabel="",
            ylabel="",
            savedir="",
            savename=None):
    if resultdata is None:
        resultdata = RESULT_SUMMER

    plt.figure(figsize=figsize, dpi=dpi)
    xpos = np.arange(len(x))

    linew = [1, 1]
    lines = ["-", "--"]
    linearr = []
    rowid = 0
    for i in datakey:
        if i == "transformer":
            label = "Trans"
        elif i == "Capsule":
            label = "Capsule"
        elif i == "resnet":
            label = "ResNet"
        elif i == "CNN":
            label = "CNN"
        elif i == "LogisticRegression":
            label = "Logic"
        else:
            label = i

        data = np.array(getdataarray(i, nums, wtl, days, p, col[0], resultdata))
        data = data.reshape(reshapedata)
        stddata = np.nanstd(data, axis=0)
        data = np.nanmean(data, axis=0)

        dmax = (data + stddata) * 100
        dmin = (data - stddata) * 100
        data = data * 100

        l, = plt.plot(xpos, data,
                      label=label + " Accuracy",
                      color=color[rowid],
                      linewidth=linew[0],
                      linestyle=lines[0], zorder=3)
        linearr.append(l)
        if fill_between[0]:
            plt.fill_between(xpos, dmin, dmax, alpha=0.1, color=color[rowid], zorder=3)

        data = np.array(getdataarray(i, nums, wtl, days, p, col[1], resultdata))
        data = data.reshape(reshapedata)
        stddata = np.nanstd(data, axis=0)
        data = np.nanmean(data, axis=0)

        dmax = (data + stddata) * 100
        dmin = (data - stddata) * 100
        data = data * 100

        l, = plt.plot(xpos, data,
                      label=label + " Recall",
                      color=color[rowid],
                      linewidth=linew[1],
                      linestyle=lines[1],
                      zorder=3)
        linearr.append(l)
        if fill_between[1]:
            plt.fill_between(xpos, dmin, dmax, alpha=0.1, color=color[rowid], zorder=3)

        rowid += 1

    markers = ["*", "v", "o", "s", "D"]
    for k in range(len(datakey)):
        for ci in range(5):
            data = np.array(getdataarray(datakey[k], nums, wtl, days, p, 3 + ci, resultdata))
            data = data.reshape(reshapedata)
            data = np.nanmean(data, axis=0)
            data = data * 100
            plt.plot(xpos, data, marker=markers[ci], color=color[k],
                     linestyle="", markersize=10, zorder=2, alpha=0.6)

    plt.xticks(xpos, x, size=16)
    plt.yticks(range(0, 101, 20), size=16)
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'

    l2 = []
    if legendarr != []:
        for l in legendarr:
            l2.append(linearr[l])
    if haslegend:
        if l2 != []:
            plt.legend(handles=l2, fontsize=16, loc=legendloc, ncol=ncol, columnspacing=1.0)
        else:
            plt.legend(fontsize=16, loc=legendloc, ncol=ncol, columnspacing=1.0)

    plt.grid(alpha=0.3)
    plt.xlabel(xlabel, fontsize=16)
    plt.ylabel(ylabel, fontsize=16)
    plt.title(title, fontsize=16)

    _savefig(savedir, savename, nums)


def drawfig2(nums=range(0, 100),
             resultdata=None,
             reshapedata=(-1, 3, 4),
             datakey=["Capsule", "resnet", "CNN"],
             x=["N", "0.75N", "0.5N", "0.25N"],
             x2=["day-1", "day-3", "day-5"],
             color=["green", "blue", "red"],
             lines=["-", "--", ":", "-."],
             linew=[1, 1, 1],
             wtl=[1],
             days=[1, 3, 5],
             p=[1, 0.75, 0.5, 0.25],
             col=[1, 2],
             fill_between=[False, False],
             figsize=(10, 6.18), dpi=100,
             haslegend=True,
             legendloc="upper right",
             ncol=1,
             legendarr=[],
             title="",
             xlabel="",
             ylabel="",
             savedir="",
             savename=None):
    if resultdata is None:
        resultdata = RESULT_SUMMER

    plt.figure(figsize=figsize, dpi=dpi)
    xpos = np.arange(len(x))
    printdaylabel = True
    linearr = []
    rowid = 0
    for i in datakey:
        if i == "transformer":
            label = "trans"
        elif i == "Capsule":
            label = "Capsule"
        elif i == "resnet":
            label = "ResNet"
        elif i == "CNN":
            label = "CNN"
        elif i == "LogisticRegression":
            label = "Logic"
        else:
            label = i

        data = np.array(getdataarray(i, nums, wtl, days, p, col[0], resultdata))
        data = data.reshape(reshapedata)
        stddata = np.nanstd(data, axis=0)
        data = np.nanmean(data, axis=0)

        dmax = (data + stddata) * 100
        dmin = (data - stddata) * 100
        data = data * 100

        dn = 0
        for j in x2:
            if printdaylabel:
                plt.text(-0.01, data[dn][0], j, ha='right', va='center',
                         fontsize=16, color=color[dn])
            l, = plt.plot(xpos, data[dn],
                          label=label,
                          color=color[dn],
                          linewidth=linew[rowid],
                          linestyle=lines[rowid],
                          zorder=3)
            linearr.append(l)
            if fill_between[0]:
                plt.fill_between(xpos, dmin[dn], dmax[dn], alpha=0.1, color=color[dn], zorder=3)
            dn += 1
        printdaylabel = False
        rowid += 1

    plt.xticks(xpos, x, size=16)
    plt.yticks(range(0, 101, 20), size=16)
    plt.xlim(xmin=-0.35, xmax=3.35)
    plt.ylim(ymin=0, ymax=101)
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'

    l2 = []
    if legendarr != []:
        for l in legendarr:
            l2.append(linearr[l])
    if haslegend:
        if l2 != []:
            plt.legend(handles=l2, fontsize=16, loc=legendloc, ncol=ncol, columnspacing=1.0)
        else:
            plt.legend(fontsize=16, loc=legendloc, ncol=ncol, columnspacing=1.0)

    plt.grid(alpha=0.3)
    plt.xlabel(xlabel, fontsize=16)
    plt.ylabel(ylabel, fontsize=16)
    plt.title(title, fontsize=16)

    _savefig(savedir, savename, nums)


def drawfigpro(resultdata=None, title1="summer",
               savename_prefix="fig", savedir=OUTDIR, datakey=None,
               color=None, ncol=2, legendarr=None, linew2=None, lines2=None):
    if resultdata is None:
        resultdata = RESULT_SUMMER
    if datakey is None:
        datakey = ["transformer", "Capsule", "resnet", "CNN"]
    if color is None:
        color = ["red", "green", "blue", "orange"]
    if legendarr is None:
        legendarr = [0, 2, 4, 6, 1, 3, 5, 7]
    if linew2 is None:
        linew2 = [1, 1, 1, 1]
    if lines2 is None:
        lines2 = ["-", "--", ":", "-."]
    drawfig(nums=range(0, 100),
            resultdata=resultdata,
            reshapedata=(-1, 5),
            datakey=datakey,
            x=["day-1", "day-2", "day-3", "day-4", "day-5"],
            color=color,
            wtl=[1],
            days=[1, 2, 3, 4, 5],
            p=[1],
            col=[1, 2],
            fill_between=[True, True],
            figsize=(10, 6.18), dpi=100,
            haslegend=True,
            legendloc="upper right",
            ncol=ncol,
            legendarr=legendarr,
            title=title1 + " z500",
            xlabel="",
            ylabel="Accuracy & Recall (%)",
            savedir=savedir,
            savename=savename_prefix + "_acc_recall")
    drawfig2(nums=range(0, 100),
             resultdata=resultdata,
             reshapedata=(-1, 3, 4),
             datakey=datakey,
             x=["N", "0.75N", "0.5N", "0.25N"],
             x2=["day-1", "day-3", "day-5"],
             color=["red", "green", "blue"],
             lines=lines2,
             linew=linew2,
             wtl=[1],
             days=[1, 3, 5],
             p=[1, 0.75, 0.5, 0.25],
             col=[1],
             fill_between=[True],
             figsize=(10, 6.18), dpi=100,
             haslegend=True,
             legendloc="upper right",
             ncol=3,
             legendarr=[dn + m * 3 for dn in range(3) for m in range(len(datakey))],
             title=title1 + " z500",
             xlabel="",
             ylabel="Accuracy (%)",
             savedir=savedir,
             savename=savename_prefix + "_acc_vs_ratio")


# ============================ Entry Point ============================
if __name__ == "__main__":
    import time
    t0 = time.time()
    print("loading summer results ...")
    RESULT_SUMMER = build_result(prepath)
    print("  keys:", {k: len(v) for k, v in RESULT_SUMMER.items()})
    print("loading winter results ...")
    RESULT_WINTER = build_result(prepathw)
    print("  keys:", {k: len(v) for k, v in RESULT_WINTER.items()})
    print("load done in %.1fs" % (time.time() - t0))

    SAVE_EXTS = [".png"]

    # Core figures (z500 only, wt=1)
    for prefix, rd, season in [("fig_summer_z500", RESULT_SUMMER, "summer"),
                               ("fig_winter_z500", RESULT_WINTER, "winter")]:
        t1 = time.time()
        drawfigpro(resultdata=rd, title1=season,
                   savename_prefix=prefix, savedir=OUTDIR)
        print("  [core] %s done in %.1fs" % (prefix, time.time() - t1))

    # z500 + t2m (wt=2) summer
    t1 = time.time()
    drawfig(nums=range(0, 100),
            resultdata=RESULT_SUMMER, reshapedata=(-1, 5),
            datakey=["transformer", "Capsule", "resnet", "CNN"],
            x=["day-1", "day-2", "day-3", "day-4", "day-5"],
            color=["red", "green", "blue", "orange"],
            wtl=[2], days=[1, 2, 3, 4, 5], p=[1],
            col=[1, 2],
            fill_between=[True, True],
            figsize=(10, 6.18), dpi=100,
            haslegend=True, legendloc="upper right", ncol=2,
            legendarr=[0, 2, 4, 6, 1, 3, 5, 7],
            title="summer z500 t2m", xlabel="", ylabel="Accuracy & Recall (%)",
            savedir=OUTDIR, savename="fig_summer_z500t2m_acc_recall")
    print("  [core] summer_z500t2m done in %.1fs" % (time.time() - t1))

    # 5-model comparison (includes LogisticRegression baseline)
    DK5 = ["transformer", "Capsule", "resnet", "CNN", "LogisticRegression"]
    CL5 = ["red", "green", "blue", "orange", "black"]
    LA5 = [0, 2, 4, 6, 8, 1, 3, 5, 7, 9]
    for prefix, rd, season in [("fig_summer_5model", RESULT_SUMMER, "summer"),
                               ("fig_winter_5model", RESULT_WINTER, "winter")]:
        t1 = time.time()
        drawfigpro(resultdata=rd, title1=season,
                   savename_prefix=prefix, savedir=OUTDIR,
                   datakey=DK5, color=CL5, ncol=2, legendarr=LA5,
                   linew2=[1, 1, 1, 1, 1],
                   lines2=["-", "--", ":", "-.", (0, (3, 1, 1, 1))])
        print("  [baseline5] %s done in %.1fs" % (prefix, time.time() - t1))

    print("ALL DONE ->", OUTDIR)