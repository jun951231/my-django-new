import pandas as pd

from admin.common.models import ValueObject, Printer, Reader


class CrimeCctvModel(object):
    vo = ValueObject()
    printer = Printer()
    reader = Reader()

    def __init__(self):

        self.crime_column = ['살인 발생, 강도 발생, 강간 발생, 절도 발생, 폭력 발생'] # Nominal
        self.arrest_column = ['살인 검거, 강도 검거, 강간 검거, 절도 검거, 폭력 검거'] # Nominal
        self.arrest_rate_column = ['살인 검거율, 강도 검거율, 강간 검거율, 절도 검거율, 폭력 검거율'] # Ratio
        self.cctv_column = ["기관명", "소계", "2013년도 이전", "2014년", "2015년", "2016년"]
        self.population_column = ['자치구', '합계', '한국인', '등록외국인', '5세이상고령자']

    def create_crime_model(self):
        vo = self.vo
        reader = self.reader
        printer = self.printer
        vo.context = 'admin/crime/data/'
        vo.fname = 'crime_in_Seoul'
        crime_file_name = reader.new_file(vo)
        print(f'파일명: {crime_file_name}')
        crime_model = reader.csv(crime_file_name)
        printer.dframe(crime_model)
        return crime_model

    def create_police_position(self):
        crime = self.create_crime_model()
        reader = self.reader
        vo = self.vo
        station_names = []
        for name in crime['관서명']:
            station_names.append('서울' + str(name[:-1] + '경찰서'))

        station_addrs = []
        station_lats = []
        station_lngs = []
        gmaps = reader.gmaps()
        for name in station_names:
            temp = gmaps.geocode(name, language='ko')
            station_addrs.append(temp[0].get('formatted_address'))
            temp_loc = temp[0].get('geometry')
            station_lats.append(temp_loc['location']['lat'])
            station_lngs.append(temp_loc['location']['lng'])
            print(f'name : {temp[0].get(f"formatted_address")}')
        gu_names = []
        for name in station_addrs:
            temp = name.split()
            gu_name = [gu for gu in temp if gu[-1] == '구'][0]
            print(f'구 이름: {gu_name}')
            gu_names.append(gu_name)
        crime['구별'] = gu_names
        print('==================================================')
        print(f"샘플 중 혜화서 정보 : {crime[crime['관서명'] == '혜화서']}")
        print(f"샘플 중 금천서 정보 : {crime[crime['관서명'] == '금천서']}")

    def create_cctv_model(self):
        vo = self.vo
        reader = self.reader
        printer = self.printer
        vo.context = 'admin/crime/data/'
        vo.fname = 'CCTV_in_Seoul'
        cctv_file_name = reader.new_file(vo)
        print(f'파일명: {cctv_file_name}')
        cctv_model = reader.csv(cctv_file_name)
        cctv_model.rename(columns={'기관명' : '구별'}, inplace= True)
        printer.dframe(cctv_model)
        cctv_model.to_csv(vo.context + 'new_data/cctv_positions.csv')
        return cctv_model

    def create_population_model(self):
        vo = self.vo
        reader = self.reader
        printer = self.printer
        vo.context = 'admin/crime/data/'
        vo.fname = 'population_in_Seoul'
        population_file_name = reader.new_file(vo)
        population_model = reader.xls(population_file_name, 2, ('B, D, G, J, N'))
        printer.dframe(population_model)
        return population_model

