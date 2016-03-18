import csv
import pandas
import sys
import Quandl as q
import matplotlib.pyplot as plt
import matplotlib.pylab
import requests
import numpy as np

quandlToken = '1bpa2GwweMigw6csBRm5'

class QuandlePlotter:
    numToPlot = 60

    def __init__(self, gdpLowToHigh, countryStats):
        self.gdpLowToHigh = gdpLowToHigh
        self.countryStats = countryStats
        DefaultSize = matplotlib.pylab.gcf().get_size_inches()
        matplotlib.pylab.gcf().set_size_inches(DefaultSize[0] * 2.5,DefaultSize[1] * 2.5)

    def grabCountryStats(self, indicators):
        numToPlot = self.numToPlot

        dne = dict()
        f = open(indicators[0], 'r+')
        for line in f:
            dne[line.replace('\n', '')] = 0

        for country in self.gdpLowToHigh:
            if numToPlot <= 0:
                break

            if country in dne.keys():
                pass
            else:
                #print countryStats[country]['name'] + '\n'
                for wwdiIndicator in indicators:
                    try:
                        requestCode = 'WWDI/' + country + '_' + wwdiIndicator
                        data = q.get(requestCode, authtoken=quandlToken)
                        # pick the latest date, and first value
                        value = data.tail(1).iloc[0, 0]
                        countryStats[country][wwdiIndicator] = value

                        numToPlot -= 1

                    except q.DatasetNotFound as e:
                        print "Dataset not found for: " + self.countryStats[country]['name']
                        f.write(country + '\n')
                        break
                    except IndexError as e:
                        print e
                    except q.ErrorDownloading as e:
                        print e


    def plotPrimarySchool(self, indicators):
        numToPlot = self.numToPlot
        x = []
        y = []
        labels = []

        for country in self.gdpLowToHigh:
            if numToPlot == 0:
                break

            stats = countryStats[country]
            if indicators[0] in stats.keys():
                x.append(stats[indicators[1]] + stats[indicators[2]])
                y.append(stats['gdp'])
                labels.append(stats['name'])
                numToPlot -= 1

        plt.scatter(x, y, marker = 'o', cmap = plt.get_cmap('Spectral'), color='blue')

        self.generateLabels(labels, x, y, 'red')

        self.saveAndClosePlot('Percentage of labor force that went to highschool')

    def generateLabels(self, labels, x, y, color):
        # Set the country labels
        for label, xVal, yVal in zip(labels, x, y):
            try:
                plt.annotate(
                    label,
                    xy=(xVal, yVal),
                    xytext=(-10, 10), textcoords = 'offset points', ha='right', va='bottom',
                    bbox=dict(boxstyle='round,pad=0.5', fc=color, alpha=0.5),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'),
                    size='x-small')

            except UnicodeDecodeError as e:
                print e
    def ploteducation(self, indicators):
        numToPlot = self.numToPlot

        colors = ['red', 'greed', 'blue']

        prim = []
        sec = []
        post = []
        countries = []

        if len(prim) != len(sec) or len(prim) != len(post):
            print 'ERROR, LENGTHS DONT MATCH!!\n'

        for country in self.gdpLowToHigh:
            if numToPlot == 0:
                break

            stats = countryStats[country]
            if indicators[0] in stats.keys():
                prim.append(stats[indicators[0]] + stats[indicators[1]] + stats[indicators[2]])
                sec.append(stats[indicators[1]] + stats[indicators[2]])
                post.append(stats[indicators[2]])
                countries.append(stats['name'])
                numToPlot -= 1

        ind = np.arange(len(countries))
        width = 0.15
        fig, ax = plt.subplots()
        rects1 = ax.bar(ind, prim, width, color='red')
        rects2 = ax.bar(ind+width, sec, width, color='y')
        rects3 = ax.bar(ind+2*width, post, width, color='green')
        ax.set_ylabel('Percentage')
        ax.set_xticks(ind + width)
        ax.set_xticklabels(countries)

        ax.legend((rects1[0], rects2[0], rects3[0]), ('Prime', 'Sec', 'Uni'))

        plt.savefig('% primary students that go on to post-secondary' + '.png', bbox_inches='tight')
        plt.show()

    def plotGdpVsQuandleCode(self, wwdiIndicator, color):
        x = []
        y = []
        labels = []
        title = "No Title"
        xLabel = "No Label"
        hasMetaData = False

        dne = dict()
        f = open(wwdiIndicator, 'r+')
        for line in f:
            dne[line.replace('\n', '')] = 0

        numToPlot = self.numToPlot
        for country in self.gdpLowToHigh:
            if numToPlot == 0:
                break

            if country in dne.keys():
                pass
            else:
                requestCode = 'WWDI/' + country + '_' + wwdiIndicator

                try:
                    data = q.get(requestCode, trim_start='2000', trim_end='2016', authtoken=quandlToken)
                    # pick the latest date, and first value
                    value = data.tail(1).iloc[0, 0]
                    labels.append(self.countryStats[country]['name'])
                    y.append(self.countryStats[country]['gdp'])
                    x.append(value)
                    numToPlot -= 1

                    if not hasMetaData:
                        # Get the MetaData
                        try:
                            meta = 'https://www.quandl.com/api/v3/datasets/' + requestCode + '/metadata.json?api_key=' + quandlToken
                            response = requests.get(meta)
                            parsedResponse = response.json()
                            title = parsedResponse['dataset']['name']
                            title = title[0:title.index(' - ')]
                            xLabel = parsedResponse['dataset']['column_names'][1]
                            hasMetaData = True;
                        except Exception as e:
                            print 'Could not get metadata for ' + self.countryStats[country]['name']


                except q.DatasetNotFound as e:
                    print "Dataset not found for: " + self.countryStats[country]['name']
                    f.write(country + '\n')
                except IndexError as e:
                    print e
                except q.ErrorDownloading as e:
                    print e


        fit = np.polyfit(x, y, 1)
        print fit
        fit_fn = np.poly1d(fit)
        plt.plot(x,y, 'x', x, fit_fn(x), '--k')
        plt.scatter(x, y, marker = 'o', cmap = plt.get_cmap('Spectral'), color=color)

        self.generateLabels(labels, x, y, color)

        plt.title(title)
        plt.xlabel(xLabel)
        plt.ylabel('GDP')

    def saveAndClosePlot(self, title):
        plt.savefig(title+ '.png', bbox_inches='tight')
        plt.close()
        DefaultSize = matplotlib.pylab.gcf().get_size_inches()
        matplotlib.pylab.gcf().set_size_inches(DefaultSize[0] * 2.5,DefaultSize[1] * 2.5)

# list is a list of strings containing only the code
def produceSingleGraphsFromList(list, title):
    for line in list:
        print 'Plotting ' + line
        qPlotter.plotGdpVsQuandleCode(line, np.random.rand(3,1))
    qPlotter.saveAndClosePlot(title)


def massProduceGraphsFromIndicatorsFile(file):
    with open(file) as indicatorsList:
        for line in indicatorsList:
            if line == 'end\n':
                print "'end' reached in indicators"
                break
            code = line.split('|')
            code = code[1].replace('\n', '')
            print 'Plotting ' + line
            qPlotter.plotGdpVsQuandleCode(code, np.random.rand(3,1))
            qPlotter.saveAndClosePlot(code)


if __name__ == "__main__":
    wwdiIndicator = 'EG_ELC_ACCS_ZS' #% of population with access
    if len(sys.argv) > 1:
        wwdiIndicator = sys.argv[1]
    else:
        print "Usage: python Main.py <WWDI Indicator> <number of countries to plot (0 for all)>"

    gdpLowToHigh = []
    countryStats = dict()
    with open('DataSets/GDP.csv', 'rb') as csvfile:
        gdpCsv = csv.reader(csvfile, delimiter=',')

        for row in gdpCsv:
            if row[0] and row[1]:
                gdpLowToHigh.append(row[0])
                countryStats[row[0]] = dict()
                countryStats[row[0]]['name'] = row[3]
                countryStats[row[0]]['gdp'] = float(row[4].replace(',',''))

    gdpLowToHigh.reverse()
    qPlotter = QuandlePlotter(gdpLowToHigh, countryStats)

    massProduceGraphsFromIndicatorsFile('DataSets/wwdi_indicators')

    # Comparing children that study and work vs only work
    #produceSingleGraphsFromList(['SL_TLF_0714_SW_ZS', 'SL_TLF_0714_WK_ZS'], "Study and Work vs Just Work")

    # % of Labor force with type of education
    eduCodes = ['SL_TLF_PRIM_ZS', 'SL_TLF_SECO_ZS',  'SL_TLF_TERT_ZS']
    #qPlotter.grabCountryStats(eduCodes)
    #qPlotter.plotPrimarySchool(eduCodes)
    #qPlotter.ploteducation(eduCodes)

    if (len(sys.argv) > 2):
        qPlotter.numToPlot = int(sys.argv[2])

