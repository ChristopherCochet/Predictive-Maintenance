import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.basemap import Basemap


class GIS_Map_Viz:

    def __init__(self, latitude_feature_name=None, longitude_feature_name=None,
                 gps_bounderies_dict={}):
        '''create a GIS gps map of the longitude and latitude values provided within the boundaries '''

        # Save data features by type and label
        print("GIS_Map_Viz: __init__ ... v2")

        self.latitude_feature_name = latitude_feature_name
        self.longitude_feature_name = longitude_feature_name
        self.gps_bounderies_dict = gps_bounderies_dict

    def _display_gps_map(self, df, labels, map_title):
        '''print the longitude & latitude GPS coordinates of the dataframe within the map bounderies'''

        print("GIS_Map_Viz: _display_gps_map ...")
        fig = plt.figure(figsize=(20, 10))
        plt.title(map_title)

        m = Basemap(
            projection='merc',
            llcrnrlat=self.gps_bounderies_dict['lat_min'] - 0.5,
            urcrnrlat=self.gps_bounderies_dict['lat_max'] + 0.5,
            llcrnrlon=self.gps_bounderies_dict['lon_min'] - 0.5,
            urcrnrlon=self.gps_bounderies_dict['lon_max'] + 0.5,
            resolution='i')

        # Reference: https://matplotlib.org/basemap/users/geography.html
        m.drawmapboundary(fill_color='#85A6D9')
        m.drawcoastlines(color='#6D5F47', linewidth=.8)
        m.drawrivers(color='green', linewidth=.4)
        m.shadedrelief()
        m.drawcountries()
        m.fillcontinents(lake_color='aqua')

        mycmap = matplotlib.colors.LinearSegmentedColormap.from_list(
            "", ["green", "yellow", "red"])
        longitudes = df[self.longitude_feature_name].tolist()
        latitudes = df[self.latitude_feature_name].tolist()
        if labels is None:
            labels = 'darkblue'
            wp = mpatches.Patch(color='darkblue', label='water points')
            plt.legend(handles=[wp], title='Location')
        else:
            labels = labels.cat.codes
            wp_functional = mpatches.Patch(
                color='green', label='water points: functional')
            wp_need_repair = mpatches.Patch(
                color='red', label='water points: non functional')
            wp_non_functional = mpatches.Patch(
                color='yellow', label='water points: functional needs repair')
            plt.legend(handles=[wp_functional, wp_need_repair,
                                wp_non_functional],
                       title='Location and Status')

        m.scatter(
            longitudes,
            latitudes,
            s=0.05,
            zorder=2,
            latlon=True,
            c=labels,
            cmap=mycmap)
        plt.show()
