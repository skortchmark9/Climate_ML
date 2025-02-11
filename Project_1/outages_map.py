import pandas as pd
import plotly.graph_objects as go
import numpy as np
from storm_data import get_storm_data, outages_bystate, storm_df


def animated_plot(storms_xr, yearly_data, storm_name):
    speeds = []
    for storm in storms_xr.wmo_wind.values:
        for timestep in storm:
            if ~np.isnan(timestep):
                speeds.append(float(timestep))

    norm_min = min(speeds)
    norm_max = max(speeds)
    
    storm = get_storm_data(storms_xr, storm_name)
    storm_season = int(storm.season[0])
    outages = outages_bystate(yearly_data, storm_season)

    # Clean Up Selected Storm Data:
    storm_data = storm_df(storms_xr, storm_name)

    # reduce outage data to duration of storm
    outages = outages.merge(storm_data[['date']], on= 'date', how = 'right')
    max_outages = outages.customers_out.max()
    
    dates = list(set(outages.date))
    dates = sorted(dates)

    # make plot
    scatter_data = []
    for i, date in enumerate(dates):
        # each timestep includes all previous hurricane locations
        for j in range(i + 1):
            scatter_data.append({
                'date': date.strftime('%Y-%m-%d'),
                'lat': storm_data['lat'][j],
                'lon': storm_data['lon'][j],
                'intensity':storm_data['intensity'][j]
        })

    scatter = pd.DataFrame(scatter_data)
    scatter.date = [d.date() for d in pd.to_datetime(scatter['date'], format='%Y-%m-%d')]

    # normalize intensity
    scatter.intensity = scatter['intensity'].transform(
        lambda x: (x - norm_min) / (norm_max - norm_min) * 50)
    scatter.intensity = scatter.intensity.ffill()

    # plot
    fig = go.Figure()
    
    # first frame
    initial_date = scatter.date[0]
    df_initial_scatter = scatter[scatter['date'] == initial_date]
    df_initial_choropleth = outages[outages['date'] == initial_date]
    
    # Choropleth layer
    fig.add_trace(go.Choropleth(
        locations=df_initial_choropleth['state_abbr'],
        z=df_initial_choropleth['customers_out'],
        locationmode='USA-states',
        colorbar_title='Customers Out of Power',
        name='Choropleth',
        zmin=0, zmax=max_outages,
        colorscale='Inferno_r'
    ))
    
    # Hurricane Path (Scattergeo)
    fig.add_trace(go.Scattergeo(
        lon=df_initial_scatter['lon'],
        lat=df_initial_scatter['lat'],
        mode='markers',
        marker=dict(color='red', symbol='circle', size=df_initial_scatter['intensity']),
        name='Hurricane Path'
    ))
    
    # Add Frames for Animation (cumulative hurricane path)
    frames = []
    for date in dates:
        df_day_scatter = scatter[scatter['date'] <= date]  # Include all previous points
        df_day_choropleth = outages[outages['date'] == date]
        
        frames.append(go.Frame(
            data=[
                go.Choropleth(locations=df_day_choropleth['state_abbr'],
                              z=df_day_choropleth['customers_out'],
                              zmin=0, zmax=max_outages),
                go.Scattergeo(lon=df_day_scatter['lon'], lat=df_day_scatter['lat'], 
                              mode='markers',
                              marker=dict(color='red', symbol='circle', size = df_day_scatter['intensity']))
            ],
            name=date.strftime('%Y-%m-%d')
        ))
    
    fig.update(frames=frames)
    
    # Add Slider & Play Button
    fig.update_layout(
        updatemenus=[{
            'buttons': [
                {'args': [None, {'frame': {'duration': 500, 'redraw': True}, 'mode': 'immediate'}],
                 'label': 'Play', 'method': 'animate'},
                {'args': [[None], {'frame': {'duration': 0, 'redraw': True}, 'mode': 'immediate'}],
                 'label': 'Pause', 'method': 'animate'}
            ],
            'direction': 'left', 'pad': {'r': 10, 't': 87}, 'showactive': False,
            'type': 'buttons', 'x': 0.1, 'xanchor': 'right', 'y': 0,'yanchor': 'top'
        }],
        sliders=[{
            'steps': [{'args': [[date], {'frame': {'duration': 0, 'redraw': True}, 'mode': 'immediate'}],
                       'label': date.strftime('%Y-%m-%d'), 'method': 'animate'} for date in scatter['date'].unique()],
            'currentvalue': {'prefix': 'Date: ', 'font': {'size': 20}}, 'x': 0.1, 'y': -0.2
        }]
    )
    
    # Layout Settings
    fig.update_layout(
        title_text='Hurricane Path with Power Outages - ' + storm_name.capitalize(),
        height=600, width = 1000
    )

    fig.update_geos(
        center=dict(lat=30, lon=-70),
        lataxis_range=[50,90], lonaxis_range=[-70, 10]
    )
        
    fig.show()
    