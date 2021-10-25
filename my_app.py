# Imports
import streamlit as st
import streamlit.components.v1 as components

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


#import streamlit as st
from PIL import Image

from pywaffle import Waffle

# Set Page Layout
img= Image.open('logo.png')
st.set_page_config(layout='wide',page_title='MS Australia survey', page_icon=img)

hide_menu_style = """
        <style>
        #MainMenu {visibility: hidden; }
        footer {visibility: hidden;}
        </style>
        """
st.markdown(hide_menu_style, unsafe_allow_html=True)



#-----------------------------------------------------------------------
#df=pd.read_csv('locationdatadFINALFULL.csv', header=[0,1])
df=pd.read_csv('data.csv', header=[0,1])

#-----------------------------------------------------------------------
# SIDEBAR
# Let's add some functionalities in a sidebar

st.sidebar.subheader('Select to Filter the data')

# By Disease status
#st.sidebar.text('People with MS')

#--------------------------------------------------------------------------------------------------------------------------------------

status = st.sidebar.radio("Connection with MS: ", ('All','Person With MS', 'Affected by MS/Carer','Professional Connection', 'Other'))
if (status == 'Person With MS'):
    #st.sidebar.success('Yes')
    dffilter = df[df['Do you have MS (or suspected MS, e.g. clinically isolated syndrome)?','Response']=='Yes']
    #map_df=df[(df['Do you have MS (or suspected MS, e.g. clinically isolated syndrome)?','Response']=='Yes') &(df['country', 'Unnamed: 186_level_1']=='Australia')][['longitude','latitude']]
elif(status == 'Affected by MS/Carer'):
    #st.success("No")
    dffilter = df[(df['FamilyMember','Response']=='Yes') | (df['Are you a paid carer for someone with MS?','Response']=='Yes')]
    #map_df=df[(df['country', 'Unnamed: 186_level_1']=='Australia')& (df['FamilyMember','Response']=='Yes') | (df['country', 'Unnamed: 186_level_1']=='Australia')&(df['Are you a paid carer for someone with MS?','Response']=='Yes')][['longitude','latitude']]
elif(status == 'All'):
    dffilter=df
    #map_df=df[df['country', 'Unnamed: 186_level_1']=='Australia'][['longitude','latitude']]
elif(status == 'Other'):
    dffilter=df[(df['FamilyMember','Response']=='No') & (df['Are you a paid carer for someone with MS?','Response']=='No') & (df['Do you have MS (or suspected MS, e.g. clinically isolated syndrome)?', 'Response']=='No')]
    #map_df=df[(df['FamilyMember','Response']=='No') & (df['Are you a paid carer for someone with MS?','Response']=='No') & (df['Do you have MS (or suspected MS, e.g. clinically isolated syndrome)?', 'Response']=='No')&(df['country', 'Unnamed: 186_level_1']=='Australia')][['longitude','latitude']]
elif(status == 'Professional Connection'):
    dffilter=df[(df.iloc[:,132].notnull())| (df.iloc[:,133].notnull())|(df.iloc[:,134].notnull())|(df.iloc[:,135].notnull())|(df.iloc[:,136].notnull())|
  (df.iloc[:,137].notnull())| (df.iloc[:,138].notnull())|(df.iloc[:,139].notnull())|(df.iloc[:,140].notnull())|(df.iloc[:,141].notnull())|
  (df.iloc[:,142].notnull())| (df.iloc[:,143].notnull())|(df.iloc[:,144].notnull())|(df.iloc[:,145].notnull())|(df.iloc[:,146].notnull())|
  (df.iloc[:,147].notnull())| (df.iloc[:,148].notnull()) ]
    #map_df=dffilter[dffilter['country', 'Unnamed: 186_level_1']=='Australia'][['longitude','latitude']]
else:
    dffilter=df
    #map_df=df[df['country', 'Unnamed: 186_level_1']=='Australia'][['longitude','latitude']]


#---------------------------------------------------------------------------------------------------------------------

#st.sidebar.text('By Sex')
filter_sex = st.sidebar.radio('Filter By Sex', options=['All', 'Female', 'Male', 'Other', 'Prefer not to say'])
if filter_sex == 'Male':
    dffilter=dffilter[dffilter['What is your gender?','Response']=='Male']
elif filter_sex == 'Female':
    dffilter=dffilter[dffilter['What is your gender?','Response']=='Female']
elif filter_sex == 'Other':
    dffilter=dffilter[dffilter['What is your gender?','Response']=='Other'] 
elif filter_sex == 'Prefer not to say':
    dffilter=dffilter[dffilter['What is your gender?','Response']=='Prefer not to say']  
else:
    pass    

#----------------------------------------------------------------------------------------------------------------------

filter_age = st.sidebar.radio('Filter By Age', options=['All','Under 18','18-30','31-50', '51-60', 'Over 60','Prefer not to say'])
if filter_age == 'Under 18':
    dffilter=dffilter[dffilter['What is your age?','Response']=='Under 18']
elif filter_age == '18-30':
    dffilter=dffilter[dffilter['What is your age?','Response']=='18-30']
elif filter_age == '31-50':
    dffilter=dffilter[dffilter['What is your age?','Response']=='31-50'] 
elif filter_age=='51-60':
    dffilter=dffilter[dffilter['What is your age?','Response']=='51-60'] 
elif filter_age=='Over 60':
    dffilter=dffilter[dffilter['What is your age?','Response']=='Over 60'] 
elif filter_age == 'Prefer not to say':
    dffilter=dffilter[dffilter['What is your age?','Response']=='Prefer not to say']  
else:
    pass    




#----------------------------------------------------------------------------------------------------------------------
if (status == 'Person With MS'):

    filter_disease_duration =st.sidebar.radio('Filter By Disease Duration', options=['All', 'Less than a year', '1-4 years', '5-10 years', '11-19 years', "More than 20 years"])
    if filter_disease_duration =='Less than a year':
            dffilter=dffilter[dffilter['How long ago were you diagnosed?', 'Response']=='Less than a year']
    elif filter_disease_duration == '1-4 years':
            dffilter=dffilter[dffilter['How long ago were you diagnosed?', 'Response']=='5-10 years']
    elif filter_disease_duration =='5-10 years':
            dffilter=dffilter[dffilter['How long ago were you diagnosed?', 'Response']=='11-19 years']
    elif filter_disease_duration =='11-19 years':
            dffilter=dffilter[dffilter['How long ago were you diagnosed?', 'Response']=='More than 20 years']
    else:
        pass
else:
    pass
#----------------------------------------------------------------------------------------------------------------------

if (status == 'Person With MS'):

    filter_disease_impact =st.sidebar.radio('Filter By Disease Impact', options=['All', 'Living Well', 'Midly affected', 'Moderately affected', 'Significant level of disability'])
    if filter_disease_impact=='Living Well':
        dffilter=dffilter[dffilter['How are you affected by your MS? (select the one that most closely matches your circumstances)', 'Response']=='I am very well (no relapses or symptoms currently)']
    elif filter_disease_impact=='Mildly affected':
        dffilter=dffilter[dffilter['How are you affected by your MS? (select the one that most closely matches your circumstances)', 'Response']=='I am mildly affected - I have very few relapses, and/or only mild symptoms or disabilities']
    elif filter_disease_impact=='Moderately affected':
        dffilter=dffilter[dffilter['How are you affected by your MS? (select the one that most closely matches your circumstances)', 'Response']=='I am moderately affected – I have occasional relapses and/or moderate level of disability or symptoms']
    elif filter_disease_impact=='Significant level of disability':
        dffilter=dffilter[dffilter['How are you affected by your MS? (select the one that most closely matches your circumstances)', 'Response']=='I have a significant level of disability and/or symptoms']    
    else:
        pass

else:
    pass

disabilitylabel_dict={'I am mildly affected - I have very few relapses, and/or only mild symptoms or disabilities': 'Mildly affected','I am moderately affected – I have occasional relapses and/or moderate level of disability or symptoms' : 'Moderately affected',
'I have a significant level of disability and/or symptoms': 'Significant level of disability','I am very well (no relapses or symptoms currently)': 'Living well'}  
disabilitylabel_order=( 'Living well','Mildly affected','Moderately affected','Significant level of disability')

#----------------------------------------------------------------------------------------------------------------------
if (status == 'Person With MS'):
    filter_disease_type=st.sidebar.radio('Filter By Disease Type', options=['All', 'Relapsing Remitting MS', 'Secondary Progressive MS', 'Primary Progressive MS', 'Clinically Isolated syndrome', 'Other', "I don't know"])
    if filter_disease_type=='Relapsing Remitting MS':
        dffilter=dffilter[dffilter['What type of MS do you have?', 'Response']=='Relapsing remitting MS']
    elif filter_disease_type=='Secondary Progressive MS':
        dffilter=dffilter[dffilter['What type of MS do you have?', 'Response']=='Secondary Progressive MS']
    elif filter_disease_type=='Primary Progressive MS':
        dffilter=dffilter[dffilter['What type of MS do you have?', 'Response']=='Primary Progressive MS']
    elif filter_disease_type=='Clinically Isolated syndrome':
        dffilter=dffilter[dffilter['What type of MS do you have?', 'Response']=='Primary Progressive MS']
    elif filter_disease_type=="I don't know":
        dffilter=dffilter[dffilter['What type of MS do you have?', 'Response']=="I don't know"]
    elif filter_disease_type=='Other':
        dffilter=dffilter[dffilter['What type of MS do you have?', 'Response']=='Other (please specify)']
    else:
        pass
else:
    pass

#----------------------------------------------------------------------------------------------------------------------

# states = st.sidebar.multiselect("State: ",
#                          ['Australian Capital Territory', 'New South Wales', 'Northern Territory', 'Queensland', 'South Australia', 'Tasmania', 'Victoria','Western Australia'])


if (status == 'Person With MS'):
    filter_states=st.sidebar.multiselect('State:',
                ['Australian Capital Territory', 'New South Wales', 'Northern Territory', 'Queensland', 'South Australia', 'Tasmania', 'Victoria','Western Australia'],
                ['Australian Capital Territory', 'New South Wales', 'Northern Territory', 'Queensland', 'South Australia', 'Tasmania', 'Victoria','Western Australia']) #pre-selected

    #dffilter=dffilter[dffilter['state', 'Unnamed: 189_level_1'].isin(filter_states)]
else:
   filter_states=st.sidebar.multiselect('State:',
                ['Australian Capital Territory', 'New South Wales', 'Northern Territory', 'Queensland', 'South Australia', 'Tasmania', 'Victoria','Western Australia'],
                ['Australian Capital Territory', 'New South Wales', 'Northern Territory', 'Queensland', 'South Australia', 'Tasmania', 'Victoria','Western Australia']) #pre-selected

dffilter=dffilter[dffilter['state', 'Unnamed: 189_level_1'].isin(filter_states)] 


#----------------------------------------------------------------------------------------------------------------------

# image = Image.open('MSRA_logo.png')
# st.image(image, width=150)
components.html("""<img src="https://www.msaustralia.org.au/wp-content/uploads/2021/09/ms-australia-full-logo.png" alt="MS Australia logo"
					style="position:absolute;top:10px;right:0;">""")

# Write a page title
st.title("MS Australia's Research Priorities Survey")

# Subheader
st.subheader('| A comprehensive person centric feedback from the MS community')
#st.subheader('| About Us')
# Another way to write text
"""
To seek feedback on our current and future research strategy, MS Australia designed and
implemented a survey with questions about the broad goals of MS research, as well as more targeted questions
about how to acheive those goals. The survey also asked about the areas of advocacy that MS Australia should
persue was wells as areas of stratgic importance. 
"""

#Separator
st.markdown('---')





#Columns Summary

st.subheader('| QUICK SUMMARY')

col1, col2, col3, col4 = st.columns(4)
# column 1
with col1:
    total = f'{int( df.shape[0] ):,}'
    st.title(total)
    st.text('PEOPLE TOOK THE SURVEY')
# # column 2
with col2:
    st.title(df[df['Do you have MS (or suspected MS, e.g. clinically isolated syndrome)?', 'Response']=='Yes'].shape[0])
    st.text('HAVE MS')


st.markdown('#')


waffle=plt.figure(
    FigureClass=Waffle,
    rows=1,
    columns=17,
    values=[1, 15],
    icons='child', icon_size=20,
    colors=sns.color_palette("deep", 2)
)
st.pyplot(waffle, use_container_width=True);

st.text('APPROXIMATELY 1 IN 16 PEOPLE WITH MS ANSWERED THE SURVEY')

st.markdown('#')








#Filters
australiafilter=df[df['country', 'Unnamed: 186_level_1']=='Australia'].index

st.markdown('---')

st.subheader('| BACIS DEMOGRAPHIC DATA')

st.markdown('#')
############################################################################################################################################

remotenessdata=dffilter['Do you live in a...', 'Response'].value_counts()

col1, col2 = st.columns(2)
with col1:
    

    map_df=dffilter[dffilter['country', 'Unnamed: 186_level_1']=='Australia'][['longitude','latitude']]
    map_df.columns=map_df.columns.droplevel(1)
    #map_df = df[['longitude', 'latitude']]
    st.map(map_df, zoom=3,use_container_width=True)


with col2:
    st.text('A BREAK DOWN OF WHERE PEOPLE LIVE')
    fig, ax = plt.subplots(1,1, figsize=(12, 6))
    ax.bar(remotenessdata.index, remotenessdata, width=0.5, 
       edgecolor='darkgray',
       linewidth=0.6,color=sns.color_palette("deep", 6))#'#d52b1e')


    fig.text(0.09, 1, 'Location', fontsize=15, fontweight='bold', font='sans')
    #fig.text(0.09, 0.95, 'The three most frequent countries have been highlighted.', fontsize=12, fontweight='light', fontfamily='sans')

    plt.box(False)

    for s in ['top', 'left', 'right']:
        ax.spines[s].set_visible(False)
    
    plt.grid(axis='y',color = 'grey', linestyle = '--', linewidth = 0.25) 

    # Tick labels

    for i in remotenessdata.index:
        ax.annotate(f"{remotenessdata[i]}", 
                   xy=(i, remotenessdata[i] + 25), #i like to change this to roughly 5% of the highest cat
                   va = 'center', ha='center',fontweight='light', fontfamily='sans')

 
    grid_y_ticks = np.arange(0, round((remotenessdata.max()+200),-1), 100) # y ticks, min, max, then step
    ax.set_yticks(grid_y_ticks)
    #ax.set_axisbelow(True)

    plt.axhline(y = 0, color = 'black', linewidth = 1.8, alpha = 0.7)

    plt.axvline(x = -0.5, color = 'black', linewidth = 1, alpha = 0.7)

    #ax.tick_params(axis='both', which='major', labelsize=12)


    ax.set_xticklabels(remotenessdata.index, fontfamily='sans', rotation=45, fontdict={'horizontalalignment':'right'});

    st.pyplot(fig, use_container_width=True)


st.markdown('#')



st.text(f'BREAK DOWN BY SEX - {status}')

#gendercounts=df[df['Do you have MS (or suspected MS, e.g. clinically isolated syndrome)?', 'Response'] == 'Yes']['What is your gender?', 'Response'].value_counts()
gendercounts=dffilter['What is your gender?', 'Response'].value_counts()
gendercounts=gendercounts.reindex(['Female', 'Male', 'Prefer not to say','Other']).fillna(0)

fig, ax = plt.subplots(1,1,figsize=(30, 2.5))
ax.set_xlim(0, gendercounts.sum()+500)
ax.set_xticks([])
ax.set_yticks([])
#fig = plt.figure(frameon=False)
fig.figsize=(10,6)
plt.barh(gendercounts.index[0], gendercounts['Female'], color=sns.color_palette("deep", 6)[0], label='Female', height=0.8,left=None)
plt.barh(gendercounts.index[0], gendercounts['Male'], left=gendercounts['Female'], color=sns.color_palette("deep", 6)[1], label='Male', height=0.8)
plt.barh(gendercounts.index[0], gendercounts['Prefer not to say'], left=gendercounts['Female']+gendercounts['Male'], color=sns.color_palette("deep", 6)[2], label='Prefer not to say', height=0.8)
plt.barh(gendercounts.index[0], gendercounts['Other'],left=gendercounts['Female']+gendercounts['Male']+gendercounts['Prefer not to say'], color=sns.color_palette("deep", 6)[3], label='Other', height=0.8)

#plt.legend()
plt.box(False)

for i in gendercounts.index:
    ax.annotate(f"{int(gendercounts[i])}", 
               xy=(gendercounts['Female']/2, i),
               va = 'center', ha='center',fontsize=40, fontweight='light', fontfamily='sans',
               color='white')
    ax.annotate("Female", 
               xy=(gendercounts['Female']/2, -0.25),
               va = 'center', ha='center',fontsize=15, fontweight='light', fontfamily='sans',
               color='white')
    
for i in range(1,2):
    ax.annotate(f"{int(gendercounts[i])}", 
              xy=(gendercounts['Female']+gendercounts['Male']/2, 0.0),
              va = 'center', ha='center',fontsize=40, fontweight='light', fontfamily='sans',
              color='white')
    ax.annotate("Male", 
              xy=(gendercounts['Female']+gendercounts['Male']/2, -0.25),#(my_df.iloc[:,0][i]+my_df.iloc[:,0][i], -0.25),
              va = 'center', ha='center',fontsize=15, fontweight='light', fontfamily='sans',
              color='white')
    
for i in range(2,3):
    ax.annotate(f"{int(gendercounts[i])}", 
               xy=(gendercounts['Female']+gendercounts['Male']+gendercounts['Prefer not to say']/2, 0.25),
               va = 'center', ha='center',fontsize=40, fontweight='light', fontfamily='sans',
               color='#1E0604')
    ax.annotate("Prefer not to say", 
             xy=(gendercounts['Prefer not to say']+gendercounts['Female']+gendercounts['Male']+50/2, -0.0),#(my_df.iloc[:,0][i]+my_df.iloc[:,0][i], -0.25),
              va = 'center', ha='center',fontsize=15, fontweight='light', fontfamily='sans',
              color='#1E0604')
    
for i in range(3,4):
    ax.annotate(f"{int(gendercounts[i])}", 
               xy=(gendercounts['Other']+gendercounts['Prefer not to say']+gendercounts['Female']+gendercounts['Male']+60/2, -0.2),
               va = 'center', ha='center',fontsize=40, fontweight='light', fontfamily='sans',
               color='#1E0604')
    ax.annotate("Other", 
               xy=(gendercounts['Other']+gendercounts['Prefer not to say']+gendercounts['Female']+gendercounts['Male']+60/2, -0.35),#(my_df.iloc[:,0][i]+my_df.iloc[:,0][i], -0.25),
               va = 'center', ha='center',fontsize=15, fontweight='light', fontfamily='sans',
               color='#1E0604')

#fig.text(0.125,1.25,'Gender of people with MS in the survey', fontfamily='san-serif',fontsize=30, fontweight='bold')
#fig.text(0.125,1.1,'We have a slightly higher preponderance of females',fontfamily='san-serif',fontsize=18)  

for s in ['top', 'left', 'right', 'bottom']:
    ax.spines[s].set_visible(False)
   
ax.legend().set_visible(False)
st.pyplot(fig, use_container_width=True)

ageorder=('Under 18','18-30','31-50', '51-60', 'Over 60','Prefer not to say')
agecounts=dffilter['What is your age?', 'Response'].value_counts(sort=False).reindex(ageorder).fillna(0)



st.markdown('#')


col1a, col2a= st.columns(2)
# column 1
with col1a:
    
    fig2, ax = plt.subplots(1,1, figsize=(12, 6))
    ax.bar(agecounts.index, agecounts, width=0.5, 
           edgecolor='darkgray',
        linewidth=0.6,color=sns.color_palette("deep", 6))#'#d52b1e')


    fig2.text(0.09, 1, 'Age', fontsize=15, fontweight='bold', fontfamily='sans')
    #fig.text(0.09, 0.95, 'The three most frequent countries have been highlighted.', fontsize=12, fontweight='light', fontfamily='sans')

    plt.box(False)

    for s in ['top', 'left', 'right']:
        ax.spines[s].set_visible(False)
        
    plt.grid(axis='y',color = 'grey', linestyle = '--', linewidth = 0.25) 

    # Tick labels

    for i in agecounts.index:
        ax.annotate(f"{agecounts[i]}", 
                       xy=(i, agecounts[i]*1.1), #i like to change this to roughly 5% of the highest cat
                       va = 'center', ha='center',fontweight='light', fontfamily='sans')

     


    grid_y_ticks = np.arange(0, (agecounts.max()*1.5), round((agecounts.max()*1.5/4),-1)) # y ticks, min, max, then step
    ax.set_yticks(grid_y_ticks)
    #ax.set_axisbelow(True)

    plt.axhline(y = 0, color = 'black', linewidth = 1.8, alpha = 0.7)

    plt.axvline(x = -0.5, color = 'black', linewidth = 1, alpha = 0.7)

    #ax.tick_params(axis='both', which='major', labelsize=12)


    ax.set_xticklabels(agecounts.index, fontfamily='sans', rotation=45, fontdict={'horizontalalignment':'right'});

    st.pyplot(fig2, use_container_width=True)

with col2a:

    st.markdown('#')
    st.markdown('#')
    st.text("THE SURVEY POPULATION IS REPSRESENTATIVE OF THE \n AUSTRALIAN MS POPULATION")


############################################################################################################################

st.subheader('| DEMOGRAPHICS OF THE MS POPULATION')


col1b, col2b= st.columns(2)

with col1b:
    fig, ax = plt.subplots(1,1, figsize=(12, 6))
    ax.bar(dffilter['What type of MS do you have?','Response'].value_counts(sort=True).index, dffilter['What type of MS do you have?','Response'].value_counts(sort=True), width=0.5, 
           edgecolor='darkgray',color=sns.color_palette("deep", 6),
           linewidth=0.6)#'#d52b1e')


    fig.text(0.09, 1, 'Types of MS', fontsize=15, fontweight='bold', fontfamily='sans')
    #fig.text(0.09, 0.95, 'The three most frequent countries have been highlighted.', fontsize=12, fontweight='light', fontfamily='sans')

    plt.box(False)

    for s in ['top', 'left', 'right']:
        ax.spines[s].set_visible(False)
        
    plt.grid(axis='y',color = 'grey', linestyle = '--', linewidth = 0.25) 

    # Tick labels

    for i in dffilter['What type of MS do you have?','Response'].value_counts(sort=True).index:
        ax.annotate(f"{dffilter['What type of MS do you have?','Response'].value_counts(sort=True)[i]}", 
                       xy=(i, dffilter['What type of MS do you have?','Response'].value_counts(sort=True)[i] + 25), #i like to change this to roughly 5% of the highest cat
                       va = 'center', ha='center',fontweight='light', fontfamily='sans')

     

    grid_y_ticks = np.arange(0, 1200, 200) # y ticks, min, max, then step
    ax.set_yticks(grid_y_ticks)
    #ax.set_axisbelow(True)

    plt.axhline(y = 0, color = 'black', linewidth = 1.8, alpha = 0.7)

    plt.axvline(x = -0.5, color = 'black', linewidth = 1, alpha = 0.7)

    #ax.tick_params(axis='both', which='major', labelsize=12)


    ax.set_xticklabels(dffilter['What type of MS do you have?','Response'].value_counts(sort=True).index, fontfamily='sans', rotation=45, fontdict={'horizontalalignment':'right'});
    st.pyplot(fig, use_container_width=True)

with col2b:
    disabilitylabel_dict={'I am mildly affected - I have very few relapses, and/or only mild symptoms or disabilities': 'Mildly affected','I am moderately affected – I have occasional relapses and/or moderate level of disability or symptoms' : 'Moderately affected',
    'I have a significant level of disability and/or symptoms': 'Significant level of disability','I am very well (no relapses or symptoms currently)': 'Living well'}  
    disabilitylabel_order=( 'Living well','Mildly affected','Moderately affected','Significant level of disability')
    fig, ax = plt.subplots(1,1, figsize=(12, 6))
    ax.bar(dffilter['How are you affected by your MS? (select the one that most closely matches your circumstances)','Response'].value_counts().rename(index=disabilitylabel_dict).reindex(disabilitylabel_order).index, dffilter['How are you affected by your MS? (select the one that most closely matches your circumstances)','Response'].value_counts().rename(index=disabilitylabel_dict).reindex(disabilitylabel_order), width=0.5, 
           edgecolor='darkgray',
           linewidth=0.6,color=sns.color_palette("deep", 6))#'#d52b1e')


    fig.text(0.09, 1, 'Disease Impact', fontsize=15, fontweight='bold', fontfamily='sans')
    #fig.text(0.09, 0.95, 'The three most frequent countries have been highlighted.', fontsize=12, fontweight='light', fontfamily='sans')

    plt.box(False)

    for s in ['top', 'left', 'right']:
        ax.spines[s].set_visible(False)
        
    plt.grid(axis='y',color = 'grey', linestyle = '--', linewidth = 0.25) 

    # Tick labels

    for i in dffilter['How are you affected by your MS? (select the one that most closely matches your circumstances)','Response'].value_counts().rename(index=disabilitylabel_dict).reindex(disabilitylabel_order).index:
        ax.annotate(f"{dffilter['How are you affected by your MS? (select the one that most closely matches your circumstances)','Response'].value_counts().rename(index=disabilitylabel_dict).reindex(disabilitylabel_order)[i]}", 
                       xy=(i, dffilter['How are you affected by your MS? (select the one that most closely matches your circumstances)','Response'].value_counts().rename(index=disabilitylabel_dict).reindex(disabilitylabel_order)[i] + 25), #i like to change this to roughly 5% of the highest cat
                       va = 'center', ha='center',fontweight='light', fontfamily='sans')

     


    grid_y_ticks = np.arange(0, 600, 100) # y ticks, min, max, then step
    ax.set_yticks(grid_y_ticks)
    #ax.set_axisbelow(True)

    plt.axhline(y = 0, color = 'black', linewidth = 1.8, alpha = 0.7)

    plt.axvline(x = -0.5, color = 'black', linewidth = 1, alpha = 0.7)

    #ax.tick_params(axis='both', which='major', labelsize=12)


    ax.set_xticklabels(dffilter['How are you affected by your MS? (select the one that most closely matches your circumstances)','Response'].value_counts().rename(index=disabilitylabel_dict).reindex(disabilitylabel_order).index, fontfamily='sans', rotation=45, fontdict={'horizontalalignment':'right'});
    st.pyplot(fig, use_container_width=True)

col1c, col2c= st.columns(2)

with col1c:
    durationorder=('Less than a year','1-4 years','5-10 years','11-19 years','More than 20 years','I don’t know')
    fig, ax = plt.subplots(1,1, figsize=(12, 6))
    ax.bar(df['How long ago were you diagnosed?','Response'].value_counts(sort=False).reindex(durationorder).index, df['How long ago were you diagnosed?','Response'].value_counts(sort=False).reindex(durationorder), width=0.5, 
           edgecolor='darkgray',
           linewidth=0.6,color=sns.color_palette("deep", 6))#'#d52b1e')


    fig.text(0.09, 1, 'Disease Duration', fontsize=15, fontweight='bold', fontfamily='sans')
    #fig.text(0.09, 0.95, 'The three most frequent countries have been highlighted.', fontsize=12, fontweight='light', fontfamily='sans')

    plt.box(False)

    for s in ['top', 'left', 'right']:
        ax.spines[s].set_visible(False)
        
    plt.grid(axis='y',color = 'grey', linestyle = '--', linewidth = 0.25) 

    # Tick labels

    for i in df['How long ago were you diagnosed?','Response'].value_counts(sort=False).reindex(durationorder).index:
        ax.annotate(f"{df['How long ago were you diagnosed?','Response'].value_counts(sort=False).reindex(durationorder)[i]}", 
                       xy=(i, df['How long ago were you diagnosed?','Response'].value_counts(sort=False).reindex(durationorder)[i] + 25), #i like to change this to roughly 5% of the highest cat
                       va = 'center', ha='center',fontweight='light', fontfamily='sans')

     

    grid_y_ticks = np.arange(0, 600, 100) # y ticks, min, max, then step
    ax.set_yticks(grid_y_ticks)
    #ax.set_axisbelow(True)

    plt.axhline(y = 0, color = 'black', linewidth = 1.8, alpha = 0.7)

    plt.axvline(x = -0.5, color = 'black', linewidth = 1, alpha = 0.7)

    #ax.tick_params(axis='both', which='major', labelsize=12)


    ax.set_xticklabels(df['How long ago were you diagnosed?','Response'].value_counts(sort=False).reindex(durationorder).index, fontfamily='sans', rotation=45, fontdict={'horizontalalignment':'right'});
    st.pyplot(fig, use_container_width=True)

    with col2c:
        st.markdown('#')
        st.text('A statistically representative sample of the Australian\nMS community when compared with AMSLS, TMPGS, NZMSP')
        

        # if (df['How long ago were you diagnosed?','Response'].value_counts(sort=False).max())<=3:
        #     st.error("There is not enough numbers in this subsection to produce statistically meaningful results")
        # else:
        #     pass

st.subheader('| RESEARCH PRIORITIES')

col1d, col2d= st.columns(2)

with col1d:
    if (df['How long ago were you diagnosed?','Response'].value_counts(sort=False).max())<=3:
             st.error("There is not enough numbers in this subsection to produce statistically meaningful results")
    else:
         pass

    #Preparing the data for research priorities graph
    df_q1=dffilter.iloc[:,9:15]
    df_q1.columns=df_q1.columns.droplevel(0)


    df_new=pd.DataFrame() #(index=[range(1,7,1)])

    for i in range(df_q1.shape[1]):
        
        df_n=pd.DataFrame(df_q1.iloc[:,i].value_counts(normalize=False).sort_index())
        #df_new=pd.merge(df_new, df_n, right_index=True, left_index=True)
        df_new=pd.concat([df_new, df_n], axis=1)
        #print(df_n)

    rank=st.slider('1=top rank, 5= lowest rank',min_value=1, max_value=5)


    df_new=df_new[['Finding a cure for MS via repair and regeneration of cells',
           'Preventing MS',
           'Better treating MS (preventing relapses and disease progression)',
           'Improving the diagnosis of MS',
           'Improving MS management and care (symptoms, rehabilitation and support)',
           "Predicting an individual's disease course (prognosis)"]]
    df_new.rename(columns={'Finding a cure for MS via repair and regeneration of cells': 'Finding a cure'}, inplace=True)

    prioritiesdata=df_new

    fig, ax = plt.subplots(1,1, figsize=(12, 5))
    if rank>=1:

        ax.bar(prioritiesdata.columns, prioritiesdata.loc[1,:],width=0.5,edgecolor='darkgray',linewidth=0.6,color=sns.color_palette("deep", 6)[0])
    else:
        pass
    if rank>=2:
        ax.bar(prioritiesdata.columns, prioritiesdata.loc[2,:],width=0.5,edgecolor='darkgray',linewidth=0.6,color=sns.color_palette("deep", 6)[1],bottom=prioritiesdata.loc[1,:])
    else:
        pass
    if rank>=3:    
        ax.bar(prioritiesdata.columns, prioritiesdata.loc[3,:],width=0.5,edgecolor='darkgray',linewidth=0.6,color=sns.color_palette("deep", 6)[2],bottom=prioritiesdata.loc[1,:]+prioritiesdata.loc[2,:])
    else:
        pass
    if rank>=4:
        ax.bar(prioritiesdata.columns, prioritiesdata.loc[4,:],width=0.5,edgecolor='darkgray',linewidth=0.6,color=sns.color_palette("deep", 6)[3],bottom=prioritiesdata.loc[1,:]+prioritiesdata.loc[2,:]+prioritiesdata.loc[3,:])
    else:
        pass
    if rank>=5:
        ax.bar(prioritiesdata.columns, prioritiesdata.loc[5,:],width=0.5,edgecolor='darkgray',linewidth=0.6,color=sns.color_palette("deep", 6)[4],bottom=prioritiesdata.loc[1,:]+prioritiesdata.loc[2,:]+prioritiesdata.loc[3,:]+prioritiesdata.loc[4,:])
    else:
        pass
    if rank>=6:
        ax.bar(prioritiesdata.columns, prioritiesdata.loc[6,:],width=0.5,edgecolor='darkgray',linewidth=0.6,color=sns.color_palette("deep", 6)[5],bottom=prioritiesdata.loc[1,:]+prioritiesdata.loc[2,:]+prioritiesdata.loc[3,:]+prioritiesdata.loc[4,:]+prioritiesdata.loc[5,:])
    else:
        pass
    #fig.text(0.09, 1, 'Priorities', fontsize=15, fontweight='bold', fontfamily='sans')

    plt.box(False)

    for s in ['top', 'left', 'right']:
        ax.spines[s].set_visible(False)
        
    plt.grid(axis='y',color = 'grey', linestyle = '--', linewidth = 0.25) 

    # Tick labels

    # for i in range(prioritiesdata.shape[0]):
    #     ax.annotate(f"{prioritiesdata[i]}", 
    #                    xy=(i, prioritiesdata[i] + 2), #i like to change this to roughly 5% of the highest cat
    #                    va = 'center', ha='center',fontweight='light', fontfamily='sans')

     


    grid_y_ticks = np.arange(0, prioritiesdata.iloc[:rank,0].sum()*1.1, round(((prioritiesdata.iloc[:,0].sum()*1.1)/10),-1)) # y ticks, min, max, then step
    ax.set_yticks(grid_y_ticks)
    #ax.set_axisbelow(True)

    plt.axhline(y = 0, color = 'black', linewidth = 1.8, alpha = 0.7)

    plt.axvline(x = -0.5, color = 'black', linewidth = 1, alpha = 0.7)

    #ax.tick_params(axis='both', which='major', labelsize=12)


    ax.set_xticklabels([s.split('(')[0] for s in prioritiesdata.columns], fontfamily='sans', rotation=45, fontdict={'horizontalalignment':'right'});


    st.pyplot(fig, use_container_width=True)


with col2d:

    st.markdown('#')
    st.markdown('#')

    ''' "PLEASE RANK THE FOLLOWING GOALS IN ORDER FROM 1 (MOST IMPORTANT) 
        TO 6 THE LEAST IMPORTANT) AND USE EACH NUMBER ONLY ONCE"'''

col1e, col2e= st.columns(2)


with col1e:

    st.markdown('#')
    st.markdown('#')

    ''' PRIORITIES RANKING DISPLAYED AS A WEIGHTED AVERAGE, GIVING EACH
        RANK A WEIGHT RESULTING IN A TOTAL SCORE FOR EACH PRIORITY'''

with col2e:

    df_new3=pd.DataFrame() 
    
    for i in range(df_q1.shape[1]):
    
        df_n=pd.DataFrame(df_q1.iloc[:,i].value_counts().sort_index(ascending=True))
        df_new3=pd.concat([df_new3, df_n], axis=1)
    
    
    df_new3['score']=[6,5,3,4,2,1]

    scores=[]
    prior=[]

    for i in range (df_new3.shape[1]-1):
        prior.append(df_new3.iloc[:,i].name)
        scores.append(((sum(df_new3.iloc[:,i]* df_new3['score']))/df_new.shape[1])/1)
    
    
    series_obj = pd.Series(scores, index=prior)

    series_obj.rename(index={'Finding a cure for MS via repair and regeneration of cells': 'Finding a cure'}, inplace=True)


    fig, ax = plt.subplots(1,1, figsize=(12, 6))

    ax.bar(series_obj.sort_values(ascending=False).index, series_obj.sort_values(ascending=False),width=0.5,edgecolor='darkgray',linewidth=0.6,color=sns.color_palette("deep", 6))

    fig.text(0.09, 1, 'WEIGHTED AVERAGE OF ALL PRIORITIES RANKED', fontsize=15, fontweight='bold', fontfamily='sans')

    plt.box(False)

    for s in ['top', 'left', 'right']:
        ax.spines[s].set_visible(False)
    
    plt.grid(axis='y',color = 'grey', linestyle = '--', linewidth = 0.25) 

    grid_y_ticks = np.arange(0, series_obj.max()*1.2, ((series_obj.max()*1.2)/4)) # y ticks, min, max, then step
    ax.set_yticks(grid_y_ticks)


    plt.axhline(y = 0, color = 'black', linewidth = 1.8, alpha = 0.7)

    plt.axvline(x = -0.5, color = 'black', linewidth = 1, alpha = 0.7)

    ax.set_xticklabels([s.split('(')[0] for s in series_obj.index], fontfamily='sans', rotation=45, fontdict={'horizontalalignment':'right'});


    st.pyplot(fig, use_container_width=True)


st.subheader('| RESEARCH STREAMS')

col1d, col2d= st.columns(2)


with col1d:

    rank2=st.slider('1=Very Important, 5= Not important',min_value=1, max_value=5)

    df_q2=dffilter.iloc[:,15:20]
    df_q2.columns=df_q2.columns.droplevel(0)

    df_new2=pd.DataFrame() #(index=[range(1,7,1)])

    for i in range(df_q2.shape[1]):
        
        df_n=pd.DataFrame(df_q2.iloc[:,i].value_counts().sort_index())
        
        df_new2=pd.concat([df_new2, df_n], axis=1)
        
    reorderlist3=('Very important', 'Important','Fairly important', "Don't know", 'Not very important', 'Not important at all')



    df_new2.fillna(0, inplace=True)
    researchstreams=df_new2.reindex(reorderlist3).T.sort_values(by='Very important', ascending=False)
    

    fig, ax = plt.subplots(1,1, figsize=(12, 6))
    if rank2>=1:

        ax.bar(researchstreams.index, researchstreams.iloc[:,0],width=0.5,edgecolor='darkgray',linewidth=0.6,color=sns.color_palette("deep", 6)[0])
    else:
        pass
    if rank2>=2:
        ax.bar(researchstreams.index, researchstreams.iloc[:,1],width=0.5,edgecolor='darkgray',linewidth=0.6,color=sns.color_palette("deep", 6)[1],bottom=researchstreams.iloc[:,0])
    else:
        pass
    if rank2>=3:    
        ax.bar(researchstreams.index, researchstreams.iloc[:,2],width=0.5,edgecolor='darkgray',linewidth=0.6,color=sns.color_palette("deep", 6)[2],bottom=researchstreams.iloc[:,0]+researchstreams.iloc[:,1])
    else:
        pass
    if rank2>=4:
        ax.bar(researchstreams.index, researchstreams.iloc[:,3],width=0.5,edgecolor='darkgray',linewidth=0.6,color=sns.color_palette("deep", 6)[3],bottom=researchstreams.iloc[:,0]+researchstreams.iloc[:,1]+researchstreams.iloc[:,2])
    else:
        pass
    if rank2>=5:
        ax.bar(researchstreams.index, researchstreams.iloc[:,4],width=0.5,edgecolor='darkgray',linewidth=0.6,color=sns.color_palette("deep", 6)[4],bottom=researchstreams.iloc[:,0]+researchstreams.iloc[:,1]+researchstreams.iloc[:,2]+researchstreams.iloc[:,3])
    else:
        pass
    if rank>=6:
        ax.bar(researchstreams.index, researchstreams.iloc[:,5],width=0.5,edgecolor='darkgray',linewidth=0.6,color=sns.color_palette("deep", 6)[5],bottom=researchstreams.iloc[:,0]+researchstreams.iloc[:,1]+researchstreams.iloc[:,2]+researchstreams.iloc[:,3]+researchstreams.iloc[:,4])
    else:
        pass

    #fig.text(0.09, 1, 'Priorities', fontsize=15, fontweight='bold', fontfamily='sans')

    plt.box(False)

    for s in ['top', 'left', 'right']:
        ax.spines[s].set_visible(False)
        
    plt.grid(axis='y',color = 'grey', linestyle = '--', linewidth = 0.25) 


    grid_y_ticks = np.arange(0, (researchstreams.max()[0]*1.1), ((researchstreams.max()[0]*1.1)/4)) # y ticks, min, max, then step
    ax.set_yticks(grid_y_ticks)
    #ax.set_axisbelow(True)

    plt.axhline(y = 0, color = 'black', linewidth = 1.8, alpha = 0.7)

    plt.axvline(x = -0.5, color = 'black', linewidth = 1, alpha = 0.7)

    #ax.tick_params(axis='both', which='major', labelsize=12)


    ax.set_xticklabels([s.split('(')[0] for s in researchstreams.index], fontfamily='sans', rotation=45, fontdict={'horizontalalignment':'right'});


    st.pyplot(fig, use_container_width=True)

with col2d:

    st.markdown('#')
    '''" QUESTION -INDICATE HOW IMPORTANT YOU FEEL EACH OF THESE RESEARCH
        STREAMS ARE TO YOU. YOU DO NOT NEED TO RANK THEM RELATIVE TO EACH
        OTHER"'''





st.subheader('| RESEARCH TYPES')

col1e, col2e= st.columns(2)



with col1e:
    st.markdown('#')

    ''' HOW MUCH PRIORITY SHOULD BE PLACED ON THE DIFFERENT TYPES OF
        RESEARCH WITHIN THIS SPECTRUM'''


with col2e:
    st.markdown('#')


    df_q3=dffilter.iloc[:,20:23]
    df_q3.columns=df_q3.columns.droplevel(0)
    df_new_q3=pd.DataFrame() #(index=[range(1,7,1)])

    for i in range(df_q3.shape[1]):
        
        df_n=pd.DataFrame(df_q3.iloc[:,i].value_counts().sort_index())
        #df_new=pd.merge(df_new, df_n, right_index=True, left_index=True)
        df_new_q3=pd.concat([df_new_q3, df_n], axis=1)


    reorderlist=('Very high priority', 'High priority', 'Medium priority',"Don't know",'Low priority', 'Not a priority')

    df_new_q3.fillna(0, inplace=True)
    df_new_q3=df_new_q3.rename(columns={"‘Basic' laboratory-based research to understand the cause and biology of MS – likely to have an impact on people with MS in the longer term (10 years or more)": 'Laboratory based research', "‘Translational' research that may develop into a clinical application within 5 years or less":'Translational Research',"‘Clinical' studies and clinical trials that are likely to have an immediate impact once the study is completed": 'Clincial Studies'})
    researchtypes=df_new_q3.reindex(reorderlist).T.sort_values(by='Very high priority', ascending=False)
    rank=5



    fig, ax = plt.subplots(1,1, figsize=(12, 6))
    if rank>=1:

        ax.bar(researchtypes.index, researchtypes.iloc[:,0],width=0.5,edgecolor='darkgray',linewidth=0.6,color=sns.color_palette("deep", 6)[0])
    else:
        pass
    if rank>=2:
        ax.bar(researchtypes.index, researchtypes.iloc[:,1],width=0.5,edgecolor='darkgray',linewidth=0.6,color=sns.color_palette("deep", 6)[1],bottom=researchtypes.iloc[:,0])
    else:
        pass
    if rank>=3:    
        ax.bar(researchtypes.index, researchtypes.iloc[:,2],width=0.5,edgecolor='darkgray',linewidth=0.6,color=sns.color_palette("deep", 6)[2],bottom=researchtypes.iloc[:,0]+researchtypes.iloc[:,1])
    else:
        pass
    if rank>=4:
        ax.bar(researchtypes.index, researchtypes.iloc[:,3],width=0.5,edgecolor='darkgray',linewidth=0.6,color=sns.color_palette("deep", 6)[3],bottom=researchtypes.iloc[:,0]+researchtypes.iloc[:,1]+researchtypes.iloc[:,2])
    else:
        pass
    if rank>=5:
        ax.bar(researchtypes.index, researchtypes.iloc[:,4],width=0.5,edgecolor='darkgray',linewidth=0.6,color=sns.color_palette("deep", 6)[4],bottom=researchtypes.iloc[:,0]+researchtypes.iloc[:,1]+researchtypes.iloc[:,2]+researchtypes.iloc[:,3])
    else:
        pass
    if rank>=6:
        ax.bar(researchtypes.index, researchtypes.iloc[:,5],width=0.5,edgecolor='darkgray',linewidth=0.6,color=sns.color_palette("deep", 6)[5],bottom=researchtypes.iloc[:,0]+researchtypes.iloc[:,1]+researchtypes.iloc[:,2]+researchtypes.iloc[:,3]+researchtypes.iloc[:,4])
    else:
        pass

    #fig.text(0.09, 1, 'RESEARCH TYPES', fontsize=15, fontweight='bold', fontfamily='sans')

    plt.box(False)

    for s in ['top', 'left', 'right']:
        ax.spines[s].set_visible(False)
        
    plt.grid(axis='y',color = 'grey', linestyle = '--', linewidth = 0.25) 



     
    grid_y_ticks = np.arange(0, (researchtypes.max()[0]*2), ((researchtypes.max()[0]*2)/4)) # y ticks, min, max, then step
    ax.set_yticks(grid_y_ticks)


    plt.axhline(y = 0, color = 'black', linewidth = 1.8, alpha = 0.7)

    plt.axvline(x = -0.5, color = 'black', linewidth = 1, alpha = 0.7)

    #ax.tick_params(axis='both', which='major', labelsize=12)


    ax.set_xticklabels([s.split('(')[0] for s in researchtypes.index], fontfamily='sans', rotation=45, fontdict={'horizontalalignment':'right'});


    st.pyplot(fig, use_container_width=True)






############################################################################################################################################
###THIS IS WHERE THE IMMEDIATE GRAPHS WILL GO

###########################################################################################################################################
st.subheader('| SYMPTOMS')

col1f, col2f= st.columns(2)



with col1f:
    rank3=st.slider('1=Very High Priority, 5= Not a priority',min_value=1, max_value=6)

    df_q10=dffilter.iloc[:,67:87]
    df_q10.columns=df_q10.columns.droplevel(0)
    df_new_q10=pd.DataFrame() 

    for i in range(df_q10.shape[1]):
        
        df_n=pd.DataFrame(df_q10.iloc[:,i].value_counts().sort_index())
        #df_new=pd.merge(df_new, df_n, right_index=True, left_index=True)
        df_new_q10=pd.concat([df_new_q10, df_n], axis=1)
        
   
    reorderlist=('Very high priority', 'High priority', 'Medium priority',"Don't know",'Low priority', 'Not a priority')

    symptomsdata=df_new_q10.reindex(reorderlist).T.sort_values('Very high priority', ascending=False)


    fig, ax = plt.subplots(1,1, figsize=(12, 6))
    if rank3>=1:

        ax.bar(symptomsdata.index, symptomsdata.iloc[:,0],width=0.5,edgecolor='darkgray',linewidth=0.6,color=sns.color_palette("deep", 6)[0])
    else:
        pass
    if rank3>=2:
        ax.bar(symptomsdata.index, symptomsdata.iloc[:,1],width=0.5,edgecolor='darkgray',linewidth=0.6,color=sns.color_palette("deep", 6)[1],bottom=symptomsdata.iloc[:,0])
    else:
        pass
    if rank3>=3:    
        ax.bar(symptomsdata.index, symptomsdata.iloc[:,2],width=0.5,edgecolor='darkgray',linewidth=0.6,color=sns.color_palette("deep", 6)[2],bottom=symptomsdata.iloc[:,0]+symptomsdata.iloc[:,1])
    else:
        pass
    if rank3>=4:
        ax.bar(symptomsdata.index, symptomsdata.iloc[:,3],width=0.5,edgecolor='darkgray',linewidth=0.6,color=sns.color_palette("deep", 6)[3],bottom=symptomsdata.iloc[:,0]+symptomsdata.iloc[:,1]+symptomsdata.iloc[:,2])
    else:
        pass
    if rank3>=5:
        ax.bar(symptomsdata.index, symptomsdata.iloc[:,4],width=0.5,edgecolor='darkgray',linewidth=0.6,color=sns.color_palette("deep", 6)[4],bottom=symptomsdata.iloc[:,0]+symptomsdata.iloc[:,1]+symptomsdata.iloc[:,2]+symptomsdata.iloc[:,3])
    else:
        pass
    if rank3>=6:
        ax.bar(symptomsdata.index, symptomsdata.iloc[:,5],width=0.5,edgecolor='darkgray',linewidth=0.6,color=sns.color_palette("deep", 6)[5],bottom=symptomsdata.iloc[:,0]+symptomsdata.iloc[:,1]+symptomsdata.iloc[:,2]+symptomsdata.iloc[:,3]+symptomsdata.iloc[:,4])
    else:
        pass

    #fig.text(0.09, 1, 'SYMPTOMS', fontsize=15, fontweight='bold', fontfamily='sans')

    plt.box(False)

    for s in ['top', 'left', 'right']:
          ax.spines[s].set_visible(False)
            
    plt.grid(axis='y',color = 'grey', linestyle = '--', linewidth = 0.25) 

         
    grid_y_ticks = np.arange(0, (symptomsdata.max()[0]*2), ((symptomsdata.max()[0]*2)/4)) # y ticks, min, max, then step
    ax.set_yticks(grid_y_ticks)


    plt.axhline(y = 0, color = 'black', linewidth = 1.8, alpha = 0.7)

    plt.axvline(x = -0.5, color = 'black', linewidth = 1, alpha = 0.7)

        #ax.tick_params(axis='both', which='major', labelsize=12)


    ax.set_xticklabels([s.split('(')[0] for s in symptomsdata.index], fontfamily='sans', rotation=45, fontdict={'horizontalalignment':'right'});


    st.pyplot(fig, use_container_width=True)  


with col2f:
    st.markdown('#')
    st.markdown('#')


    ''' THE INDICATED PRIORITY TO BE PLACED ON RESERACH INTO TREATING
        OR MANAGING DIFFERENT MS SYMPTOMS'''

st.markdown('#')
st.markdown('#')
st.subheader('| TERMINOLOGY')

col1g, col2g= st.columns(2)


with col1g:
    st.markdown('#')
    st.markdown('#')
    

    '''LANGUAGE USAGE IS EVERY IMPORTANT TO EMPOWER PEOPLE WITH MS
       AND TO REDUCE BIASES- THEREFORE WE ASKED THE COMMUNITY TO
       RANK TERMS USED TO DESCRIBE SOMEONE WITH MS'''



with col2g:
    st.markdown('#')
    st.markdown('#')
    

    rank4=st.slider('1=Top Rank, 5= Lowest Rank',min_value=1, max_value=5)

    termology_df=dffilter.iloc[:,158:163]
    termology_df.columns=termology_df.columns.droplevel(0)

    term_df=pd.DataFrame()

    for i in range(termology_df.shape[1]):
        term_df=pd.concat([term_df,termology_df.iloc[:,i].value_counts().to_frame()], axis=1)

    
    fig, ax = plt.subplots(1,1, figsize=(12, 6))
    if rank4>=1:
        ax.bar(term_df.columns, term_df.iloc[0,:],width=0.5,edgecolor='darkgray',linewidth=0.6,color=sns.color_palette("deep", 6)[0])
    else:
        pass
    if rank4>=2:
        ax.bar(term_df.columns, term_df.iloc[1,:],width=0.5,edgecolor='darkgray',linewidth=0.6,color=sns.color_palette("deep", 6)[1],bottom=term_df.iloc[0,:])
    else:
        pass
    if rank4>=3:    
        ax.bar(term_df.columns, term_df.iloc[2,:],width=0.5,edgecolor='darkgray',linewidth=0.6,color=sns.color_palette("deep", 6)[2],bottom=term_df.iloc[0,:]+term_df.iloc[1,:])
    else:
        pass
    if rank4>=4:
        ax.bar(term_df.columns, term_df.iloc[3,:],width=0.5,edgecolor='darkgray',linewidth=0.6,color=sns.color_palette("deep", 6)[3],bottom=term_df.iloc[0,:]+term_df.iloc[1,:]+term_df.iloc[2,:])
    else:
        pass
    if rank4>=5:
        ax.bar(term_df.columns, term_df.iloc[4,:],width=0.5,edgecolor='darkgray',linewidth=0.6,color=sns.color_palette("deep", 6)[4],bottom=term_df.iloc[0,:]+term_df.iloc[1,:]+term_df.iloc[2,:]+term_df.iloc[3,:])
    else:
        pass
    


    #fig.text(0.09, 1, 'TERMINOLOGY', fontsize=15, fontweight='bold', fontfamily='sans')
    #fig.text(0.09, 0.95, 'The three most frequent countries have been highlighted.', fontsize=12, fontweight='light', fontfamily='sans')

    plt.box(False)

    for s in ['top', 'left', 'right']:
        ax.spines[s].set_visible(False)
        
    plt.grid(axis='y',color = 'grey', linestyle = '--', linewidth = 0.25) 

    # Tick labels

    # for i in term_df.T.shape[0]:
    #     ax.annotate(f"{term_df.T[i]}", 
    #                    xy=(i, term_df.T[i] + 2), #i like to change this to roughly 5% of the highest cat
    #                    va = 'center', ha='center',fontweight='light', fontfamily='sans')

     


    grid_y_ticks = np.arange(0,((term_df.iloc[0,:].sum())*1.3), (((term_df.iloc[0,:].sum())*1.1)/4)) # y ticks, min, max, then step
    ax.set_yticks(grid_y_ticks)
    #ax.set_axisbelow(True)

    plt.axhline(y = 0, color = 'black', linewidth = 1.8, alpha = 0.7)

    plt.axvline(x = -0.5, color = 'black', linewidth = 1, alpha = 0.7)

    #ax.tick_params(axis='both', which='major', labelsize=12)


    ax.set_xticklabels(term_df.columns, fontfamily='sans', rotation=45, fontdict={'horizontalalignment':'right'});

    st.pyplot(fig, use_container_width=True)



col1h, col2h= st.columns(2)

with col1h:
    st.markdown('#')
    st.markdown('#')
    st.subheader('| ADVOCACY')


    df_adv=dffilter.iloc[:,116:127]
    df_adv.columns=df_adv.columns.droplevel(0)
    
    df_advocacy=pd.DataFrame() #(index=[range(1,7,1)])

    for i in range(df_adv.shape[1]):
        
        df_n=pd.DataFrame(df_adv.iloc[:,i].value_counts().sort_index())
        #df_new=pd.merge(df_new, df_n, right_index=True, left_index=True)
        df_advocacy=pd.concat([df_advocacy, df_n], axis=1)
    

    df_advocacy=df_advocacy.reindex(reorderlist)

    
    df_advocacy=df_advocacy[df_advocacy.iloc[0,:].sort_values().index]
    df_advocacy.rename(columns={'Increase in MS educational resources for people with MS, their families, carers and the general public.':'Increase in MS educational resources','Improved access to telehealth (e.g. video links for neurologist consultations).':'Improved access to telehealth', "Access to transport, appropriate housing (including better residential care), employment services, and other concessions to improve the quality of day to day life of people with MS.":'concessions to improve the quality of day to day life of people with MS' ,'Improved access to assistive technologies for day to day life (aids and equipment)':'Improved access to assistive technologies'}, inplace=True)


    fig, ax = plt.subplots(1,1, figsize=(12, 6))
    ax.bar(df_advocacy.columns, df_advocacy.iloc[0,:],width=0.5,edgecolor='darkgray',linewidth=0.6,color=sns.color_palette("deep", 6)[0])
    ax.bar(df_advocacy.columns, df_advocacy.iloc[1,:],width=0.5,edgecolor='darkgray',linewidth=0.6,color=sns.color_palette("deep", 6)[1],bottom=df_advocacy.iloc[0,:])
    ax.bar(df_advocacy.columns, df_advocacy.iloc[2,:],width=0.5,edgecolor='darkgray',linewidth=0.6,color=sns.color_palette("deep", 6)[2],bottom=df_advocacy.iloc[0,:]+df_advocacy.iloc[1,:])
    ax.bar(df_advocacy.columns, df_advocacy.iloc[3,:],width=0.5,edgecolor='darkgray',linewidth=0.6,color=sns.color_palette("deep", 6)[3],bottom=df_advocacy.iloc[0,:]+df_advocacy.iloc[1,:]+df_advocacy.iloc[2,:])
    ax.bar(df_advocacy.columns, df_advocacy.iloc[4,:],width=0.5,edgecolor='darkgray',linewidth=0.6,color=sns.color_palette("deep", 6)[4],bottom=df_advocacy.iloc[0,:]+df_advocacy.iloc[1,:]+df_advocacy.iloc[2,:]+df_advocacy.iloc[3,:])
    #ax.bar(df_advocacy.columns, df_advocacy.iloc[6,:],width=0.5,edgecolor='darkgray',linewidth=0.6,color=sns.color_palette("deep", 6)[5],bottom=df_advocacy.T.loc[1,:]+df_advocacy.T.loc[2,:]+df_advocacy.T.loc[3,:]+df_advocacy.T.loc[4,:]+df_advocacy.T.loc[5,:])


    #fig.text(0.09, 1, 'ADVOCACY', fontsize=15, fontweight='bold', fontfamily='sans')
    #fig.text(0.09, 0.95, 'The three most frequent countries have been highlighted.', fontsize=12, fontweight='light', fontfamily='sans')

    plt.box(False)

    for s in ['top', 'left', 'right']:
        ax.spines[s].set_visible(False)
        
    plt.grid(axis='y',color = 'grey', linestyle = '--', linewidth = 0.25) 

    # Tick labels

    # for i in df_advocacy.T.shape[0]:
    #     ax.annotate(f"{df_advocacy.T[i]}", 
    #                    xy=(i, df_advocacy.T[i] + 2), #i like to change this to roughly 5% of the highest cat
    #                    va = 'center', ha='center',fontweight='light', fontfamily='sans')

     


    grid_y_ticks = np.arange(0,((df_advocacy.iloc[:,0].sum())*1.3), (((df_advocacy.iloc[:,0].sum())*1.1)/4)) # y ticks, min, max, then step
    ax.set_yticks(grid_y_ticks)
    #ax.set_axisbelow(True)

    plt.axhline(y = 0, color = 'black', linewidth = 1.8, alpha = 0.7)

    plt.axvline(x = -0.5, color = 'black', linewidth = 1, alpha = 0.7)

    #ax.tick_params(axis='both', which='major', labelsize=12)


    ax.set_xticklabels(df_advocacy.columns, fontfamily='sans', rotation=45, fontdict={'horizontalalignment':'right'});

    st.pyplot(fig, use_container_width=True)
    



with col2h:
    st.markdown('#')
    st.markdown('#')
    st.markdown('#')
    

    '''AREAS OF ADVOCACY THAT ARE IMPORTANT TO THE
       MS COMMUNITY'''
