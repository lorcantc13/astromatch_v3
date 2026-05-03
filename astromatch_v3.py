import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# --- 1. APP CONFIG ---
st.set_page_config(page_title="AstroMatch V2", layout="wide")

# --- 2. DATA LOADING ---
@st.cache_data
def load_data():
    try:
        analogues = pd.read_csv('analogues_v2.csv')
        targets = pd.read_csv('targets_v2.csv')
        return analogues, targets
    except FileNotFoundError:
        st.error("Data files not found. Ensure 'analogues_v2.csv' and 'targets_v2.csv' are in the directory.")
        return pd.DataFrame(), pd.DataFrame()

analogues_df, targets_df = load_data()

# --- 3. SCORING ENGINE (GAUSSIAN-JACCARD) ---
def calculate_suitability(site_min, site_max, target_min, target_max):
    if pd.isna(site_min) or pd.isna(target_min) or pd.isna(site_max) or pd.isna(target_max):
        return None # Indicates missing data
    
    s_mid = (site_min + site_max) / 2
    t_mid = (target_min + target_max) / 2
    t_width = max(target_max - target_min, 1.0)
    
    sigma = t_width 
    gaussian = np.exp(-((s_mid - t_mid)**2) / (2 * sigma**2))
    
    overlap_min = max(site_min, target_min)
    overlap_max = min(site_max, target_max)
    
    if overlap_max > overlap_min:
        intersection = overlap_max - overlap_min
        union = max(site_max, target_max) - min(site_min, target_min)
        return max(gaussian, intersection / union)
    return gaussian

# --- 4. SIDEBAR: WEIGHTING & VISUALS ---
st.sidebar.header("🎯 Importance Weights")
st.sidebar.info("Toggle parameters and adjust influence (1-10)")

params_config = {
    "Temperature": {"color": "#EF553B", "default": 5, "col_prefix": "T"},
    "Salinity": {"color": "#00CC96", "default": 5, "col_prefix": "Sal"},
    "pH": {"color": "#636EFA", "default": 5, "col_prefix": "pH"},
    "Pressure": {"color": "#AB63FA", "default": 5, "col_prefix": "Pres"},
    "Isolation": {"color": "#FFA15A", "default": 5, "col_prefix": "Iso", "help": "Derived from Physical Confinement, Hydrologic Connectivity, and Isolation Time rubrics."},
    "Redox": {"color": "#19D3F3", "default": 5, "col_prefix": "Redox", "help": "Derived from Evidence of Reductants, Oxidants, and Metabolic Equilibrium rubrics."}
}

user_weights = {}
active_params = []

for name, info in params_config.items():
    st.sidebar.markdown(f"**<span style='color:{info['color']}'>{name}</span>**", unsafe_allow_html=True)
    
    if "help" in info:
        st.sidebar.caption(f"ℹ️ {info['help']}")
        
    is_on = st.sidebar.toggle(f"Include {name}", value=True, key=f"tog_{name}")
    val = st.sidebar.slider(f"{name} Weight", 1, 10, info['default'], label_visibility="collapsed", key=f"sld_{name}", disabled=not is_on)
    
    if is_on:
        user_weights[name] = val
        active_params.append(name)

# Dynamic Donut Chart
if user_weights:
    weights_df = pd.DataFrame({
        "Parameter": list(user_weights.keys()),
        "Weight": list(user_weights.values())
    })
    fig_donut = px.pie(
        weights_df, values='Weight', names='Parameter', hole=0.5, color='Parameter',
        color_discrete_map={k: v['color'] for k, v in params_config.items()}
    )
    fig_donut.update_layout(showlegend=False, height=220, margin=dict(t=10, b=10, l=10, r=10), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    st.sidebar.plotly_chart(fig_donut, use_container_width=True)
else:
    st.sidebar.warning("Please enable at least one parameter.")

# --- 5. MAIN INTERFACE ---
st.title("🌌 AstroMatch MCDA Tool v2")

col_a, col_b = st.columns(2)
with col_a:
    body_choice = st.selectbox("1. Select Target Body", ["Enceladus", "Europa", "Mars"])

with col_b:
    if body_choice == "Enceladus" and not targets_df.empty:
        env_list = targets_df['Target_env'].unique().tolist()
        target_env = st.selectbox("2. Select Environment", env_list)
    else:
        st.selectbox("2. Select Environment", ["🚧 Coming Soon..."], disabled=True)
        target_env = None

# --- ADVANCED OPTIONS & CUSTOM UPLOAD ---
custom_sites_df = pd.DataFrame()

with st.expander("🛠 Advanced Options"):
    st.selectbox("Select Organism (Preset Weights)", ["🚧 Coming Soon..."], disabled=True)
    st.divider()
    
    st.markdown("**Import Custom Analogue Site**")
    st.info("Upload a CSV with your own analogue data. It will be temporarily added to the analysis for this session.")
    
    # Generate the expected template structure dynamically
    template_cols = ['Site', 'lat', 'lon']
    for p_info in params_config.values():
        prefix = p_info['col_prefix']
        if prefix in ['Iso', 'Redox']:
            template_cols.extend([f"{prefix}_score", f"{prefix}_rel"])
        else:
            template_cols.extend([f"{prefix}_min", f"{prefix}_max", f"{prefix}_rel"])
            
    template_df = pd.DataFrame(columns=template_cols)
    
    st.download_button(
        label="📥 Download CSV Template",
        data=template_df.to_csv(index=False).encode('utf-8'),
        file_name='astromatch_custom_template.csv',
        mime='text/csv',
    )
    
    uploaded_file = st.file_uploader("Upload Completed Template", type=["csv"])
    
    if uploaded_file:
        try:
            custom_sites_df = pd.read_csv(uploaded_file)
            custom_sites_df['User_Supplied'] = True # Tag for flagging system
            st.success(f"Successfully loaded {len(custom_sites_df)} custom site(s)!")
        except Exception as e:
            st.error(f"Error reading file: {e}")

st.divider()

# --- 6. EXECUTION & OUTPUT ---
if st.button("🚀 Run Analysis") and target_env and user_weights:
    target_data = targets_df[targets_df['Target_env'] == target_env].iloc[0]
    results = []
    
    # Combine original data with custom uploaded data
    working_analogues = analogues_df.copy()
    if not custom_sites_df.empty:
        working_analogues = pd.concat([working_analogues, custom_sites_df], ignore_index=True)
    
    w_sum_total = sum(user_weights.values())

    for _, site in working_analogues.iterrows():
        site_fits = {}
        site_rels = {}
        active_site_weights = {}
        flags = []
        
        # Check if site is user supplied
        if site.get('User_Supplied') == True:
            flags.append("⚠️ User-Supplied Data")
        
        # Calculate Fits and Handle Missing Data
        for param in active_params:
            prefix = params_config[param]['col_prefix']
            
            min_col = f"{prefix}_min" if f"{prefix}_min" in site else f"{prefix}_score"
            max_col = f"{prefix}_max" if f"{prefix}_max" in site else f"{prefix}_score"
            rel_col = f"{prefix}_rel"
            
            t_min_col = f"{prefix}_min" if f"{prefix}_min" in target_data else f"{prefix}_score"
            t_max_col = f"{prefix}_max" if f"{prefix}_max" in target_data else f"{prefix}_score"

            fit = calculate_suitability(site.get(min_col), site.get(max_col), target_data.get(t_min_col), target_data.get(t_max_col))
            
            if fit is not None:
                site_fits[param] = fit
                site_rels[param] = site.get(rel_col, 1) # Default to low rel if missing
                active_site_weights[param] = user_weights[param]
            else:
                weight_pct = user_weights[param] / w_sum_total
                if weight_pct > 0.05:
                    flags.append(f"⚠️ Missing {param} data")
                site_fits[param] = "N/A"
                site_rels[param] = "N/A"

        actual_w_sum = sum(active_site_weights.values())
        if actual_w_sum > 0:
            final_score = sum(site_fits[p] * active_site_weights[p] for p in active_site_weights) / actual_w_sum
        else:
            final_score = 0.0
            
        if active_site_weights:
            fits_for_conf = [site_fits[p] for p in active_site_weights]
            rels_for_conf = [site_rels[p] for p in active_site_weights]
            
            if sum(fits_for_conf) > 0:
                conf_score = sum(f * r for f, r in zip(fits_for_conf, rels_for_conf)) / sum(fits_for_conf)
            else:
                conf_score = np.mean(rels_for_conf)
                
            for p in active_site_weights:
                if site_fits[p] > 0.7 and site_rels[p] == 1 and (active_site_weights[p]/actual_w_sum) > 0.2:
                    flags.append(f"⚠️ Low Rel on core driver: {p}")
        else:
            conf_score = 0.0
            
        alert_str = " | ".join(set(flags)) if flags else "✅ Reliable"

        res_dict = {
            "Site": site['Site'],
            "Suitability": round(final_score, 4),
            "Confidence": round(conf_score, 2),
            "Alerts": alert_str,
            "lat": site.get('lat', None),
            "lon": site.get('lon', None)
        }
        
        for p in params_config.keys():
            res_dict[f"{p} Fit"] = site_fits.get(p, "Off/NA")
            res_dict[f"{p} Rel"] = site_rels.get(p, "Off/NA")
            
        results.append(res_dict)

    res_df = pd.DataFrame(results).sort_values("Suitability", ascending=False).reset_index(drop=True)
    res_df.index += 1 

    st.session_state['res_df'] = res_df
    st.session_state['target_env'] = target_env

 # --- Profile Layout ---
    st.write("### Analogue Footprint vs Target")
    
    # 1. RADAR CHART (Full Width / Prominent)
    categories = active_params
    r_vals = []
    
    # 1. Convert the Pandas Series to a pure Python dictionary
    site_dict = site_data.to_dict()
    
    for p in categories:
        col_name = f"{p} Fit"
        
        # 2. Safely get the value (defaults to 0 if the column is completely missing)
        val = site_dict.get(col_name, 0)
        
        # 3. Verify it is a valid number and not NaN
        if pd.notna(val) and isinstance(val, (int, float, np.number)):
            r_vals.append(float(val))
        else:
            r_vals.append(0)
            
    fig_radar = go.Figure()
    fig_radar.add_trace(go.Scatterpolar(r=[1]*len(categories), theta=categories, fill='toself', name='Target', line_color='gold'))
    fig_radar.add_trace(go.Scatterpolar(r=r_vals, theta=categories, fill='toself', name=selected_site, line_color='cyan'))
    
    # Make it taller and move the legend to the bottom so it doesn't squash the chart
    fig_radar.update_layout(
        height=500, 
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])), 
        showlegend=True, 
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
        margin=dict(t=40, b=40, l=40, r=40)
    )
    st.plotly_chart(fig_radar, use_container_width=True, key="radar")

    st.divider()

    # 2. TABLE AND MAP (Side-by-Side underneath)
    c1, c2 = st.columns(2)
    
    with c1:
        st.write("### Parameter Breakdown")
        breakdown_data = []
        for p in active_params:
            breakdown_data.append({
                "Parameter": p,
                "Fidelity": site_data[f"{p} Fit"],
                "Data Quality (Rel)": site_data[f"{p} Rel"]
            })
        st.dataframe(pd.DataFrame(breakdown_data), use_container_width=True, hide_index=True)

    with c2:
        st.write("### Global Location")
        if pd.notna(site_data['lat']) and pd.notna(site_data['lon']):
            map_df = pd.DataFrame({"lat": [site_data['lat']], "lon": [site_data['lon']], "Site": [selected_site]})
            fig_map = px.scatter_geo(map_df, lat="lat", lon="lon", hover_name="Site", projection="natural earth")
            fig_map.update_traces(marker=dict(size=12, color="red"))
            fig_map.update_geos(showcountries=True, countrycolor="RebeccaPurple")
            fig_map.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=300)
            st.plotly_chart(fig_map, use_container_width=True, key="map")
        else:
            st.warning("No coordinate data (lat/lon) available for this site.")
